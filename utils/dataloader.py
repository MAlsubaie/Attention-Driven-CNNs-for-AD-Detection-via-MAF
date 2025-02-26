import math
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed

class CustomDataGenerator(tf.keras.utils.Sequence):
    ''' Custom DataGenerator to load images 
    
    Arguments:
        data_frame = pandas data frame with filenames and labels
        batch_size = divide data in batches
        shuffle = shuffle data before loading
        img_shape = image shape in (h, w, d) format
        augmentation = data augmentation to make model robust to overfitting
    
    Output:
        Img: numpy array of image
        label : output binary label for image
    '''
    
    def __init__(self, dataframe, batch_size=4, img_shape=(128, 128, 128, 1)):
        self.df = dataframe
        self.image_paths = self.df['ADNI_path'].values
        self.labels = self.df['Group'].values

        # Binary classification mapping (e.g., 'CN' as 0 and 'AD' as 1)
        self.label_names = {'CN': 0, 'AD': 1}  # Only binary classes
        self.labels_binary = self.df['Group'].map(self.label_names).values
        print("*"*34)
        print(f"{len(self.image_paths)} Images found with binary classification.")
        print(f"Samples per class: {dict(zip(*np.unique(self.labels, return_counts=True)))}")
        print("*"*34)

        self.batch_size = batch_size
        self.img_shape = img_shape
        self.max_workers = 4

    def __len__(self):
        ''' return total number of batches '''
        return math.ceil(len(self.image_paths) / self.batch_size)

    def on_epoch_end(self):
        ''' shuffle data after every epoch '''
        indices = np.arange(len(self.image_paths))
        np.random.shuffle(indices)

        self.image_paths = self.image_paths[indices]
        self.labels_binary = self.labels_binary[indices]

    def __data_augmentation(self, img):
        ''' function for applying data augmentation '''
        pass

    def resize_image(self, image, new_size):
        resample = sitk.ResampleImageFilter()
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetSize(new_size)
        resample.SetOutputSpacing([sz*spc/nsz for sz, spc, nsz in zip(image.GetSize(), image.GetSpacing(), new_size)])
        resample.SetInterpolator(sitk.sitkLinear)
        return resample.Execute(image)

    def __process_image(self, idx):
        ''' process a single image '''
        try:
            image_path = self.image_paths[idx]
            label = self.labels_binary[idx]

            # Load image
            image = sitk.ReadImage(image_path, sitk.sitkFloat32)

            # Resize image to (128, 128, 128)
            resized_image = self.resize_image(image, self.img_shape)

            # Convert to numpy array
            resized_image_np = sitk.GetArrayFromImage(resized_image)
            resized_image_np = np.transpose(resized_image_np, (2, 1, 0))

            # Normalize image
            resized_image_np = (resized_image_np - np.mean(resized_image_np)) / np.std(resized_image_np)

            # Convert to numpy array
            image_tensor = np.array(resized_image_np, dtype=np.float32)

            return image_tensor, np.array(label, dtype=np.float32)
        
        except Exception as e:
            print("Exception in processing image:", e)
            return None, None

    def __getitem__(self, idx):
        batch_x = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        indices = np.where(np.isin(self.image_paths, batch_x))[0]

        images, labels = [], []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.__process_image, file_id) for file_id in indices]
            for future in as_completed(futures):
                img, lbl = future.result()
                if img is not None and lbl is not None:
                    images.append(img)
                    labels.append(lbl)

        images = np.array(images).reshape(-1, self.img_shape[0], self.img_shape[1], self.img_shape[2], 1)
        labels = np.array(labels)
        
        return images, labels
