#basic imports
#---------------------------------------------------------------------
import constant
import os
import sys
import numpy as np
sys.path.append(constant.MRCNN_DIR)
#======================================================================

#Everything mrcnn
#------------------------------------------------------------------------
from mrcnn import utils
#=========================================================================


class DDSMDataset(utils.Dataset):
    """
    Dataset class for training cancer detection on
    Digital Database for Screening Mammography.
	
	Attributes
	--------------

	"""


    def __init__(self, num_of_images, image_dir, orig_height, orig_width):
        super().__init__(self)
        
        """
		Parameters
		-------------
		num_of_images: int

		image_dir: str
			path to the folder containing data and mask subfolders
		
		orig_height: int
			height of the image

		orig_width: int
			width of the image
        """

        # Add classes
        self.add_class('cancer', 1, 'mass')  #takes source, class_id, class_name
   		
        # add images 
        for i, fp in enumerate(image_fps):
            img_path = os.path.join(image_dir, )


    def image_reference(self, image_id):
    	info = self.image_info[image_id]
        return info['path']


    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image


     def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                if a['Target'] == 1:
                    x = int(a['x'])
                    y = int(a['y'])
                    w = int(a['width'])
                    h = int(a['height'])
                    mask_instance = mask[:, :, i].copy()
                    cv2.rectangle(mask_instance, (x, y), (x+w, y+h), 255, -1)
                    mask[:, :, i] = mask_instance
                    class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)   



#sanity check
#-------------------------------------------------------------------------
if __name__=="__main__":
	dataset_train = DDSMDataset(num_of_images, image_dir, orig_height, orig_width)
	dataset_test = DDSMDataset(num_of_images, image_dir, orig_height, orig_width)
