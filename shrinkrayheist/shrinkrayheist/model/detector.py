import random

import cv2
from PIL import Image, ImageDraw, ImageFont


def _label_to_color(label):
    random.seed(label)  # Use label as seed to generate a stable color
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)


class Detector:
    def __init__(self, yolo_dir="/root/yolo", from_tensor_rt=True, threshold=0.5):
        # local import
        from ultralytics import YOLO
        cls = YOLO
        
        self.threshold = threshold
        self.yolo_dir = yolo_dir
        if from_tensor_rt:
            self.model = cls(f"{self.yolo_dir}/yolo11n.engine", task="detect")
        else:
            self.model = cls(f"{self.yolo_dir}/yolo11n.pt", task="detect")
    
    def to(self, device):
        self.model.to(device)

    def predict(self, img, silent=True):
        """
        Note: img can be any of the following:
            Union[str, pathlib.Path, int, PIL.Image.Image, list, tuple, numpy.ndarray, torch.Tensor]

            Batch not supported.
            
        Runs detection on a single image and returns a list of
        ((xmin, ymin, xmax, ymax), class_label) for each detection
        above the given confidence threshold.
        """
        results = list(self.model(img, verbose=not silent, conf=self.threshold))[0]
        boxes = results.boxes

        predictions = []
        # Iterate over the bounding boxes
        for xyxy, conf, cls_idx in zip(boxes.xyxy, boxes.conf, boxes.cls):
            if conf.item() >= self.threshold:
                # Convert bounding box tensor to Python floats
                x1, y1, x2, y2 = xyxy.tolist()
                # Map class index to class label using model/ results
                label = results.names[int(cls_idx.item())]
                predictions.append(((x1, y1, x2, y2), label))
        
        #convert original image to rgb
        original_image = results.orig_img
        cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB, original_image)
        
        return dict(predictions=predictions, original_image=original_image)
    
    def set_threshold(self, threshold):
        """
        Sets the confidence threshold for predictions.
        """
        self.threshold = threshold

    def draw_box(self, img, predictions, draw_all=False):
        """
        Draw bounding boxes on 'img'.

        :param img: The image to draw on (PIL.Image or NumPy array).
        :param predictions: A list of ((xmin, ymin, xmax, ymax), label).
        :param draw_all: If True, draw *all* bounding boxes.
                         If False, draw only the first one.
        :return: A PIL image with boxes and labels drawn.
        """
        if not predictions:
            return img  # No detections, return as is

        # Convert to PIL.Image if needed
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        draw = ImageDraw.Draw(img)

        min_dim = min(img.width, img.height)
        scale_factor = (
            min_dim / 600.0
        )

        line_width = max(
            1, int(4 * scale_factor)
        )
        font_size = max(10, int(20 * scale_factor))
        text_offset = font_size + 3

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        print(f"Labels: {[x[-1] for x in predictions]}")

        if draw_all:
            for (x1, y1, x2, y2), label in predictions:
                color = _label_to_color(label)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
                draw.text((x1, y1 - text_offset), label, fill=color, font=font)
        else:
            (x1, y1, x2, y2), label = predictions[0]
            color = _label_to_color(label)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            draw.text((x1, y1 - text_offset), label, fill=color, font=font)

        return img
    
    def id2name(self, i):
        """
        Converts a class index to a class name.
        """
        return self.model.names[i]
    
    @property
    def classes(self):
        return self.model.names
        
    
def demo():
    import os
    model = Detector()
    model.set_threshold(0.5)
    
    img_path = f"{os.path.dirname(__file__)}/../../media/minion.png" 
        
    img = Image.open(img_path)
    results = model.predict(img)
    
    predictions = results["predictions"]
    original_image = results["original_image"]
        
    out = model.draw_box(original_image, predictions, draw_all=True)
    
    save_path = f"{os.path.dirname(__file__)}/demo_output.png"
    out.save(save_path)
    print(f"Saved demo to {save_path}!")

if __name__ == '__main__':    
    demo()
