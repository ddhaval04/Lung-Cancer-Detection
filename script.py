#!/usr/bin/env python -W ignore::DeprecationWarning
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import dicom
import glob
from matplotlib import pyplot as plt
import os
import cv2
import mxnet as mx
import pandas as pd
from sklearn import cross_validation
import scipy.ndimage
from skimage import measure, morphology
from sklearn.cluster import KMeans
from skimage.transform import resize
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, dilation, binary_erosion, binary_closing
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi
# import xgboost as xgb
import tensorflow as tf
import numpy as np
import gc
import db
import argparse



IMG_SIZE_PX = 306
SLICE_COUNT = 20

n_classes = 2
batch_size = 10
x = tf.placeholder('float')
y = tf.placeholder('float')
keep_rate = 0.8

def connect_to_db():
    sqlite_file = "lung_cancer_NEW.sqlite" # name of the sqlite database file
    conn = db.connect(sqlite_file)
    return conn


def get_extractor():
    print("Loading Resnet-50 Model ...")
    model = mx.model.FeedForward.load('model/resnet-50', 0, ctx=mx.cpu(), numpy_batch_size=1)
    fea_symbol = model.symbol.get_internals()["flatten0_output"]
    feature_extractor = mx.model.FeedForward(ctx=mx.cpu(), symbol=fea_symbol, numpy_batch_size=64,
                                             arg_params=model.arg_params, aux_params=model.aux_params,
                                             allow_extra_params=True)

    return feature_extractor


def read_data_from_db(cursor, table_name_1, col_names):
    col_names = ','.join(col_names)
    print("Reading data from DB ...")
    cursor.execute("SELECT {col} from {tn}".format(col = col_names, tn = table_name_1))
    data = cursor.fetchall()

    return data


def normalize_hu(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.

    return image


def read_data(path, labels_path, output_path, cursor = 0):
    # path = r"F:\Data Science\DS Bowl\stage1_sample"
    print("Reading Data ...")
    patients = os.listdir(path)
    patients.sort()
    labels_df = pd.read_csv(labels_path, index_col=0)
    labels_df.head()
    # net = get_extractor()
    count = 0
    much_data = []
    target_dir = output_path
    for folder in glob.glob(path + "/*"):
        # count += 1
        # if count == 3:
            # break
        print("Processing patient no.: ", count + 1)
        count = count + 1
        lstFilesDCM = []  # create an empty list
        masked_lung = []
        patient_name = folder.split('\\')[-1]
        try:
            label = labels_df.get_value(patient_name, 'cancer')
            for s in os.listdir(folder):
                if ".dcm" in s.lower():
                    lstFilesDCM.append(os.path.join(folder,s))
            # print("Loading Images ...")
            scans = load_images(lstFilesDCM)
            # print("Done ...")
            images = get_pixels_hu(scans)
            # print(images.shape)
            pixel_spacing = scans[0].PixelSpacing
            pixel_spacing.append(scans[0].SliceThickness)
            # images = rescale_patient_images(images, pixel_spacing, 1.0)
            for i in range(images.shape[0]):
                patient_dir = target_dir + "/" + patient_name + "/"
                if not os.path.exists(patient_dir):
                    os.mkdir(patient_dir)
                img_path = patient_dir + "img_" + str(i).rjust(4, '0') + "_i.png"
                org_img = images[i]
                # if there exists slope,rotation image with corresponding degree
                # if cos_degree > 0.0:
                    # org_img = cv_flip(org_img,org_img.shape[1],org_img.shape[0],cos_degree)
                img, mask = get_segmented_lungs(org_img.copy())
                org_img = normalize_hu(org_img)
                # print("Image Shape: ", org_img.shape)
                # print("Mask Shape: ", mask.shape)
                cv2.imwrite(img_path, org_img * 255)
                cv2.imwrite(img_path.replace("_i.png", "_m.png"), mask * 255)
            # pix_resampled, spacing = resample(images, scans, [1, 1, 1])
            # x, y, z = pix_resampled.shape
            # pix_resampled = pix_resampled.reshape((x, y, z, 1))
            # feats = net.predict(pix_resampled)
            # print(feats.shape)
            # np.save(output_path + "folder.npy", feats)
            # for img in pix_resampled:
                # print("learning features ...")
                # feats = net.predict(images)
                # print(feats.shape)
                # np.save(output_path + "folder.npy", feats)
                # masked_lung.append(make_lungmask(img))
            # feats = net.predict(pix_resampled)
            # masked_lung = np.array(masked_lung)
            # print(masked_lung.shape)
            # exit()
            # if label == 1: label = np.array([0,1])
            # elif label == 0: label = np.array([1,0])
            # print(masked_lung.shape)
            # exit()
            # cursor.execute("insert into lung_masks values (?, ?, ?)", (patient_name, masked_lung, label))
            # print("Data Inserted !")
            # much_data.append([masked_lung,label])
        except KeyError as e:
            print("This is unlabelled data!!")


def load_patient_images(patient_id, base_dir=None, wildcard = "*.*"):
    if base_dir == None:
        base_dir = r"F:\Data Science\DS Bowl\Extracted_Images\\"
    src_dir = base_dir + patient_id + "\\"
    src_img_paths = glob.glob(src_dir + wildcard)
    src_img_paths = [im for im in src_img_paths]
    src_img_paths.sort()
    images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in src_img_paths]
    # print(np.array(images).shape)
    images = [im.reshape((1, ) + im.shape) for im in images]
    # x, y, z = images.shape
    res = np.vstack(images)
    print(res.shape)
    return res


def get_segmented_lungs(im):
    # Step 1: Convert into a binary image.
    binary = im < -400
    # Step 2: Remove the blobs connected to the border of the image.
    cleared = clear_border(binary)
    # Step 3: Label the image.
    label_image = label(cleared)
    # Step 4: Keep the labels with 2 largest areas.
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    # Step 5: Erosion operation with a disk of radius 2. This operation is seperate the lung nodules attached to the blood vessels.
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    # Step 6: Closure operation with a disk of radius 10. This operation is    to keep nodules attached to the lung wall.
    selem = disk(10)
    binary = binary_closing(binary, selem)
    # Step 7: Fill in the small holes inside the binary mask of lungs.
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    # Step 8: Superimpose the binary mask on the input image.
    get_high_vals = binary == 0
    im[get_high_vals] = -2000
    return im, binary


def load_images(file_path):
    # print(len(file_path))
    slices = [dicom.read_file(s) for s in file_path]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

    
def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)    
    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    resize_factor = spacing / new_spacing
    # print(image.shape)
    # print(resize_factor)
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing


def make_lungmask(img, display=False):
    '''
    We have to make sure that we set our threshold between the lung pixel values and the denser tissue pixel values.
    To do this, we reset the pixels with the minimum value to the average pixel value near the center of the picture
    and perform kmeans clustering with k=2. 
    This seems to work well for both scenarios.
    '''
    row_size= img.shape[0]
    col_size = img.shape[1]
    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img / std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size / 5) : int(col_size / 5 * 4), int(row_size / 5) : int(row_size / 5 * 4)] 
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))

    labels = measure.label(dilation) # Different labels are displayed in different colors
    label_vals = np.unique(labels)
#     print(labels[0])
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0

    #
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation

    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask*img, cmap='gray')
        ax[2, 1].axis('off')
        
        plt.show()
    return mask*img


def rescale_patient_images(images_zyx, org_spacing_xyz, target_voxel_mm, is_mask_image=False, verbose=False):
    if verbose:
        print("Spacing: ", org_spacing_xyz)
        print("Shape: ", images_zyx.shape)

    # print "Resizing dim z"
    resize_x = 1.0
    resize_y = float(org_spacing_xyz[2]) / float(target_voxel_mm)
    interpolation = cv2.INTER_NEAREST if is_mask_image else cv2.INTER_LINEAR
    res = cv2.resize(images_zyx, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)  # opencv assumes y, x, channels umpy array, so y = z pfff
    # print "Shape is now : ", res.shape

    res = res.swapaxes(0, 2)
    res = res.swapaxes(0, 1)
    # print "Shape: ", res.shape
    resize_x = float(org_spacing_xyz[0]) / float(target_voxel_mm)
    resize_y = float(org_spacing_xyz[1]) / float(target_voxel_mm)

    # cv2 can handle max 512 channels..
    if res.shape[2] > 512:
        res = res.swapaxes(0, 2)
        res1 = res[:256]
        res2 = res[256:]
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res1 = cv2.resize(res1, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
        res2 = cv2.resize(res2, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res = numpy.vstack([res1, res2])
        res = res.swapaxes(0, 2)
    else:
        res = cv2.resize(res, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)

    # channels = cv2.split(res)
    # resized_channels = []
    # for channel in  channels:
    #     channel = cv2.resize(channel, dsize=None, fx=resize_x, fy=resize_y)
    #     resized_channels.append(channel)
    # res = cv2.merge(resized_channels)
    # print "Shape after resize: ", res.shape
    res = res.swapaxes(0, 2)
    res = res.swapaxes(2, 1)
    if verbose:
        print("Shape after: ", res.shape)
    return res


def train_xgboost():
    df = pd.read_csv('data/stage1_labels.csv')
    print(df.head())

    x = np.array([np.mean(np.load('stage1/%s.npy' % str(id)), axis=0) for id in df['id'].tolist()])
    y = df['cancer'].as_matrix()

    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y,
                                                                   test_size=0.20)

    clf = xgb.XGBRegressor(max_depth=10,
                           n_estimators=1500,
                           min_child_weight=9,
                           learning_rate=0.05,
                           nthread=8,
                           subsample=0.80,
                           colsample_bytree=0.80,
                           seed=4242)

    clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=50)
    return clf


def prepare_image_for_net3D(img):
    img = img.astype(np.float32)
    img -= 41
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    return img


if __name__ == '__main__':
    path = r"F:\Data Science\DS Bowl\stage1"
    labels_path = r"F:\Data Science\DS Bowl\stage1_labels.csv"
    output_path = r"E:\Dhaval\Data Science\Projects\Deep Learning\Lung_Cancer\Output\Extracted_Images"
    table_name_1 = "lung_masks"
    col_names = ["Patient_id TEXT", "Masks array", "Label array"]
    data = []
    parser = argparse.ArgumentParser()
    parser.add_argument("Option", nargs = "?", choices = ["P", "CNN", "B"], default = "P", help = "Select 'P': Pre-process the CT-scan images for each patient 'CNN': Apply Convolutional Neural Network on the pre-processed data from database 'B': Perform pre-processing and CNN both")
    args = parser.parse_args()
    # conn = connect_to_db()
    # cursor = conn.cursor()
    # db.create_table(cursor, table_name_1, col_names)
    if args.Option == "P":
        read_data(path, labels_path, output_path)
        # db.disconnect(conn)
    elif args.Option == "CNN":
        # data = read_data_from_db(cursor, table_name_1, col_names)
        # print(data[0][1].shape)
        # patient_id = "00cba091fa4ad62cc3200a657aeb957e"
        net = get_extractor()
        patient_dir = ".\Output"
        images_input_path = r"F:\Data Science\DS Bowl\Extracted_Images\\"
        for folder in glob.glob(path + '/*'):
            patient_id = folder.split('\\')[-1]
            patient_dir = patient_dir + "\\" + patient_id
            if not os.path.exists(patient_dir):
                os.mkdir(patient_dir)
            im = load_patient_images(patient_id, base_dir = images_input_path, wildcard = "*_i.png")
            im = prepare_image_for_net3D(im)
            print(im.shape)
            feats = net.predict(im)
            print(feats.shape)
            np.save(patient_dir, feats)
        
        # train_data = data
        # validation_data = data
        # train_neural_network(x, validation_data)
        # db.disconnect(conn)
    elif args.Option == "B":
        read_data(path, labels_path, output_path)
        # db.disconnect(conn)
    # cursor.execute("select * from {tn}".format(tn = table_name_1))
    # results = cursor.fetchall()
    # print("Number of rows: ",len(results))
    # exit()
    # calc_features()
    # make_submit()