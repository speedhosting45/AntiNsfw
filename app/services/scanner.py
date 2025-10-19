import asyncio
import hashlib
import time
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageFilter
import cv2
import io
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import logging
from ..models.schemas import NSFWCategory, ScanResponse

logger = logging.getLogger(__name__)

class AdvancedNudityScanner:
    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.is_loaded = False
        self.total_scans = 0
        
    async def load_models(self):
        """Load ML models asynchronously"""
        try:
            # Load pre-trained feature extractor
            base_model = EfficientNetB4(
                weights='imagenet',
                include_top=False,
                input_shape=(380, 380, 3)
            )
            
            # Add custom classification head
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(512, activation='relu')(x)
            predictions = Dense(len(NSFWCategory), activation='softmax')(x)
            
            self.model = Model(inputs=base_model.input, outputs=predictions)
            
            # Feature extractor for detailed analysis
            self.feature_extractor = Model(
                inputs=base_model.input,
                outputs=base_model.get_layer('top_activation').output
            )
            
            self.is_loaded = True
            logger.info("NSFW models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    async def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Advanced image preprocessing"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize while maintaining aspect ratio
            target_size = (380, 380)
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            image_array = np.array(image) / 255.0
            
            # Expand dimensions for batch
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise

    async def extract_features(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Extract advanced image features"""
        features = {}
        
        try:
            # Get deep features
            deep_features = self.feature_extractor.predict(image_array, verbose=0)
            features['deep_features'] = deep_features.flatten().tolist()
            
            # Color analysis
            image_rgb = (image_array[0] * 255).astype(np.uint8)
            hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            
            # Skin tone detection
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
            features['skin_ratio'] = float(skin_ratio)
            
            # Edge density
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size
            features['edge_density'] = float(edge_density)
            
            # Color histogram features
            hist_r = cv2.calcHist([image_rgb], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([image_rgb], [1], None, [32], [0, 256])
            hist_b = cv2.calcHist([image_rgb], [2], None, [32], [0, 256])
            
            features['color_histogram'] = {
                'red': hist_r.flatten().tolist(),
                'green': hist_g.flatten().tolist(),
                'blue': hist_b.flatten().tolist()
            }
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            
        return features

    async def generate_heatmap(self, image_array: np.ndarray, predictions: np.ndarray) -> Optional[bytes]:
        """Generate attention heatmap for visualization"""
        try:
            # Grad-CAM like heatmap generation
            last_conv_layer = self.feature_extractor.get_layer('top_activation')
            grad_model = tf.keras.models.Model(
                [self.feature_extractor.inputs],
                [last_conv_layer.output, self.feature_extractor.output]
            )
            
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image_array)
                class_output = predictions[:, np.argmax(predictions[0])]
            
            grads = tape.gradient(class_output, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
            
            # Resize heatmap to original image size
            heatmap = cv2.resize(heatmap, (380, 380))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Convert to bytes
            _, buffer = cv2.imencode('.png', heatmap)
            return buffer.tobytes()
            
        except Exception as e:
            logger.error(f"Heatmap generation failed: {e}")
            return None

    async def scan_image(self, image_data: bytes, detailed: bool = False, 
                        heatmap: bool = False) -> ScanResponse:
        """Main scanning function"""
        start_time = time.time()
        
        try:
            if not self.is_loaded:
                await self.load_models()

            # Preprocess image
            image_array = await self.preprocess_image(image_data)
            
            # Generate image hash for caching
            image_hash = hashlib.md5(image_data).hexdigest()
            
            # Make prediction
            predictions = self.model.predict(image_array, verbose=0)[0]
            
            # Get confidence scores for each category
            category_scores = {
                category: float(predictions[i]) 
                for i, category in enumerate(NSFWCategory)
            }
            
            # Determine if NSFW
            nsfw_confidence = max(category_scores.values())
            is_nsfw = nsfw_confidence > 0.7  # Configurable threshold
            
            # Generate detailed analysis
            detailed_analysis = None
            if detailed:
                detailed_analysis = await self.extract_features(image_array)
                detailed_analysis['prediction_confidence'] = nsfw_confidence
            
            # Generate heatmap
            heatmap_data = None
            if heatmap:
                heatmap_data = await self.generate_heatmap(image_array, predictions)
            
            processing_time = time.time() - start_time
            self.total_scans += 1
            
            return ScanResponse(
                is_nsfw=is_nsfw,
                confidence=nsfw_confidence,
                categories=category_scores,
                processing_time=processing_time,
                image_hash=image_hash,
                heatmap_url=f"/heatmaps/{image_hash}.png" if heatmap_data else None,
                detailed_analysis=detailed_analysis,
                warnings=[]
            )
            
        except Exception as e:
            logger.error(f"Image scanning failed: {e}")
            processing_time = time.time() - start_time
            return ScanResponse(
                is_nsfw=False,
                confidence=0.0,
                categories={},
                processing_time=processing_time,
                warnings=[f"Processing error: {str(e)}"]
            )

# Global scanner instance
scanner = AdvancedNudityScanner()
