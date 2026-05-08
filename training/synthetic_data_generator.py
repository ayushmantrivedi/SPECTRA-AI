"""
SPECTRA Synthetic Data Generation Pipeline
==========================================
Generates 100K samples of Image + HSG pairs for training Expert Heads.
Simulates a Blender/Unity headless render pipeline.
"""

import os
import json
import random
import uuid
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
from PIL import Image, ImageDraw

class SyntheticSurgicalDatasetGenerator:
    def __init__(self, output_dir: str = "datasets/synthetic_hsg_pairs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Meta-categories for randomization
        self.labels = ["liver", "gallbladder", "artery", "tissue", "clamp", "scalpel", "suction"]
        self.materials = ["organic", "metal", "latex", "fluid"]
        self.lighting_types = ["ambient", "spotlight", "diffuse"]

    def generate_random_hsg(self) -> Dict[str, Any]:
        """Generates a randomized Huffman Scene Graph structure."""
        scene_id = str(uuid.uuid4())[:8]
        num_children = random.randint(3, 8)
        
        children = []
        for i in range(num_children):
            label = random.choice(self.labels)
            # Randomized bbox [x, y, w, h]
            w = random.randint(50, 200)
            h = random.randint(50, 200)
            x = random.randint(0, 512 - w)
            y = random.randint(0, 512 - h)
            
            child = {
                "id": f"{label}_{i}_{scene_id}",
                "label": label,
                "weight": round(random.random(), 4),
                "depth": 1,
                "bbox": [x, y, w, h],
                "attributes": {
                    "color": {
                        "primary": random.choice(["red", "pink", "white", "silver"]),
                        "hex": "#{:06x}".format(random.randint(0, 0xFFFFFF))
                    },
                    "edge": {
                        "boundary_type": random.choice(["hard", "soft", "shadow"]),
                        "sharpness": round(random.random(), 2)
                    },
                    "texture": {
                        "material_class": random.choice(self.materials),
                        "roughness": round(random.random(), 2)
                    }
                }
            }
            children.append(child)
            
        return {
            "id": "SCENE_ROOT",
            "label": "surgical_scene",
            "weight": 1.0,
            "depth": 0,
            "global_attributes": {
                "lighting": random.choice(self.lighting_types),
                "palette": ["#FF0000", "#C0C0C0"],
                "ambient_intensity": round(random.uniform(0.4, 0.9), 2)
            },
            "children": children
        }

    def render_simulated_image(self, hsg: Dict[str, Any], filename: str):
        """
        Simulates the Blender render by drawing the HSG onto a canvas.
        In a real scenario, this would trigger a Blender/bpy subprocess.
        """
        img = Image.new("RGB", (512, 512), color=(20, 10, 10)) # Dark surgical background
        draw = ImageDraw.Draw(img)
        
        for child in hsg["children"]:
            bbox = child["bbox"]
            color = child["attributes"]["color"]["hex"]
            # Draw a simple shape to represent the object
            draw.rectangle([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]], 
                          fill=color, outline="white")
            draw.text((bbox[0], bbox[1]), child["label"], fill="white")
            
        img.save(self.output_dir / filename)

    def run(self, num_samples: int = 100):
        print(f"[SyntheticGenerator] Starting generation of {num_samples} samples...")
        
        for i in range(num_samples):
            hsg = self.generate_random_hsg()
            img_name = f"sample_{i:06d}.png"
            json_name = f"sample_{i:06d}.json"
            
            self.render_simulated_image(hsg, img_name)
            
            with open(self.output_dir / json_name, "w") as f:
                json.dump(hsg, f, indent=2)
                
            if (i + 1) % 10 == 0:
                print(f"Generated {i+1}/{num_samples}...")

        print(f"[SUCCESS] Synthetic dataset ready at {self.output_dir}")

if __name__ == "__main__":
    generator = SyntheticSurgicalDatasetGenerator()
    # Running a small batch for demonstration; production target is 100k
    generator.run(num_samples=50)
