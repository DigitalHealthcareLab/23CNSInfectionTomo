import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from pathlib import Path
import h5py
import skimage
import cv2
from scipy.ndimage import center_of_mass, zoom
import math

def draw_bbox(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bbox = [cv2.boundingRect(contour) for contour in contours]
    x, y, w, h = bbox[0]
    return x, y, w, h

def norm(img):
    return (img - img.min()) / (img.max() - img.min())

def make_255(img):
    return (norm(img)*255).astype(np.uint8)

def generate_masks(img):

    # Change RI images into uint8 type
    img_gray = make_255(img)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    
    # Bilateral filter (smoothing keeping edges)
    object_size = min(30, max(1, np.mean(img_gray.shape) / 40))
    sigma_range = 0.1
    sigma = object_size / 2.35
    blur = skimage.restoration.denoise_bilateral(
        image=img_gray,
        sigma_color=sigma_range,
        sigma_spatial=sigma,
    )
    blur = make_255(blur)

    # OTSU thresholding
    ret, thres = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # fill holes in RI image by morphologyex operation
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel, iterations=3)

    # Setting backgrounds by closing function
    sure_bg = cv2.dilate(morph, kernel, iterations=3)
    
    # Distance transform to set 100% forgeround
    dist_transform = cv2.distanceTransform(morph, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)

    # Setting unknown area between sure_bg and sure_fg
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Set Markers
    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed
    markers = cv2.watershed(img_rgb, markers)

    return markers

def generate_contour(img, denoise=False, threshold=1.34):
    z, _, _ = img.shape
    mask_3d = np.zeros(img.shape)
    kernel = np.ones((3, 3), np.uint8)
    
    for i in np.arange(z):
        mask = (img[i] > threshold).astype(np.uint8)
        if denoise == True:
            mask = cv2.fastNlMeansDenoising(mask)
        mask_3d[i] = mask
    return mask_3d


class PhysicalQuantity():
    def __init__(self, tcf_path):
        self.path = tcf_path
        with h5py.File(self.path, 'r') as f:
            self.ri = f['Data/3D/000000'][()]
            self.resx = f['Data/3D'].attrs['ResolutionX'][()][0]   # um/pixel
            self.resy = f['Data/3D'].attrs['ResolutionY'][()][0]   # um/pixel
            self.resz = f['Data/3D'].attrs['ResolutionZ'][()][0]   # um/pixel
        self.voxel_volume = self.resx*1000 * self.resy*1000 * self.resz*1000    # nm^3
        self.medium_ri = 1.337
        self.margin = 20
        self.threshold = 1.345
        self.denoise = True
        self._init()

    def _generate_bboxes(self, margin=20):
        rimip = self.ri.max(0)
        bboxes = []
        patches = []
        for val in range(2, self.markers.max()):
            mask = (self.markers==val).astype(np.uint8)
            x, y, w, h = draw_bbox(mask)
            if x-margin < 0:
                x = margin
            if y-margin < 0:
                y = margin
            if x+w+margin > rimip.shape[1]:
                x = x - (x + w + margin - rimip.shape[1])
            if y+h+margin > rimip.shape[0]:
                y = y - (y + h + margin - rimip.shape[0])
            
            bboxes.append([x-margin, y-margin, w+2*margin, h+2*margin])
        self.bboxes = bboxes

    def _patching(self):
        patches = []
        kernel = np.ones((3, 3), np.uint8)
        for i, val in enumerate(range(2, self.markers.max())):
            mask = (self.markers==val).astype(np.uint8)
            x, y, w, h = self.bboxes[i]
            # patch = self.ri * mask
            patch = self.ri[..., y:y+h, x:x+w]
            # patch[patch == 0] = self.medium_ri * 10000
            patches.append(patch)
        self.patches = patches
            
    def _draw_bboxes(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(self.ri.max(0), vmin=13300, vmax=14000, cmap='gray')
        for i, bbox in enumerate(self.bboxes):
            x, y, w, h = bbox
            ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor='red'))
            ax.text(x-10, y-10, i, fontsize=15, fontweight='semibold', color='red')
        plt.tight_layout()
        plt.savefig(f"{self.path.name}.png", dpi=150)
        plt.close()

    def _init(self):
        self.markers = generate_masks(self.ri.max(0))
        self._generate_bboxes()
        self._patching()

    def __getitem__(self, idx):
        '''
        (1) volume: mask 내부의 voxel의 개수와 각 voxel의 부피를 곱함
        (2) surface area: mask의 isosurface를 구성하는 삼각형 면들의 면적을 모두 합산
        (3) sphericity: volume과 surface area로부터 측정함: π^(1/3)·(6·volume)^(2/3)·1/surface area
        (4) dry mass density: RI 와 dry mass density 간의 선형관계로부터 voxel마다 구하고, mask내에서 평균 냄
        (5) dry mass: voxel별 dry mass density를 mask 내의 부피에서 적분
        (6) meanRI: RI mean value
        '''
        patch = self.patches[idx]
        patch = patch / 10000.0 # changing integer into float
        patch = zoom(patch, [self.resz/self.resy, 1, 1])

        mask_3d = generate_contour(patch, denoise=self.denoise, threshold=self.threshold)
        
        # volume (fL)
        volume_dimless = np.sum(mask_3d)
        volume = volume_dimless * (self.resx*1000)**3 * 1.0e-9 #nm^3 -> fL

        # surface area (um^2)
        vert, faces, normals, values = skimage.measure.marching_cubes(mask_3d)
        surface_area_dimless = skimage.measure.mesh_surface_area(vert, faces)
        surface_area = surface_area_dimless * self.resy * self.resx   # um2

        # sphericity
        sphericity = math.pi ** (1/3) * (6 * volume_dimless) ** (2/3) / surface_area_dimless

        # drymass density (g/dL)
        p_density = (patch.copy() - self.medium_ri) / 0.2 # g/mL
        p_density = p_density * mask_3d
        p_density = np.sum(p_density) / np.sum(mask_3d)
        p_density = p_density * 100 # g/dL

        # drymass (pg)
        drymass = p_density * volume # g/dL * fL = 1.0e-14 g/L
        drymass = drymass * 1.0e-2 # 1.0e-14 g/L -> 1.0e-12 g/L -> pg

        # meanRI
        meanri = patch.mean()
        
        return volume, p_density, drymass, meanri, surface_area, sphericity

    def __len__(self):
        return len(self.bboxes)
    

if __name__ == "__main__":
    #tcf_path = Path('./20220707.150135.868.2nd_CD3CD8-001.TCF')
    tcf_path = Path('TCF/20221107.122408.047.CD8-001.TCF')

    print(f"{tcf_path.name} is read.")
    quantity_calculator = PhysicalQuantity(tcf_path)
    print("Initialization is done.")

    # if you want to manually set patches, add patches to "cells.patches"
    # cells.bboxes = manual_bboxes
    # cells.patches = manual_patches
    
    # view extracted bboxes
    print("Rule-based object detection is done. png file saved")
    quantity_calculator._draw_bboxes()
    
    # save calculation result as csv
    print("Calculationg physical quantities...")
    df = pd.DataFrame(columns=["filename", "idx", "volume", "p_density", "drymass", "meanri", "surface_area", "sphericity"])
    print(quantity_calculator)
    for i, cell_instance in enumerate(quantity_calculator):
        print(f"{i} / {len(quantity_calculator)}")
        volume, p_density, drymass, meanri, surface_area, sphericity = cell_instance
        row = pd.Series({"filename":tcf_path.name, 
                            "idx":i, 
                            "volume":volume, 
                            "p_density":p_density, 
                            "drymass":drymass, 
                            "meanri":meanri, 
                            "surface_area":surface_area, 
                            "sphericity":sphericity})
        df = pd.concat([df, row.to_frame().T], ignore_index=True)
        df.to_csv(f"{tcf_path.name}.csv", index=True)

