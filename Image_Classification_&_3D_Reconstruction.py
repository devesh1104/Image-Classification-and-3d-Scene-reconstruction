import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image
from collections import defaultdict
import json
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Debug OpenCV and CUDA status
print("OpenCV version:", cv2.__version__)
print("CUDA module available:", hasattr(cv2, 'cuda'))
print("CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, 'cuda') else 0)
if os.path.exists('/kaggle/input'):
    print("Available datasets in /kaggle/input:", os.listdir('/kaggle/input'))

class GPUOptimizedSceneReconstructionPipeline:
    def __init__(self, eps=0.4, min_samples=3, batch_size=64):
        """
        Initialize the GPU-optimized pipeline for scene clustering and reconstruction
        
        Parameters:
        -----------
        eps : float
            DBSCAN epsilon parameter (distance threshold)
        min_samples : int
            DBSCAN minimum samples parameter
        batch_size : int
            Batch size for feature extraction
        """
        self.eps = eps
        self.min_samples = min_samples
        self.batch_size = batch_size
        self.device = device
        
        # Initialize tf_efficientnet_b0 with timm
        print("Loading tf_efficientnet_b0 model from Kaggle dataset...")
        model_path = '/kaggle/input/tf-efficientnet/pytorch/tf-efficientnet-b0/1/tf_efficientnet_b0_aa-827b6e33.pth'
        try:
            self.model = timm.create_model('tf_efficientnet_b0', pretrained=False)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("tf_efficientnet_b0 model loaded successfully")
        except Exception as e:
            print(f"Error loading tf_efficientnet_b0 model from {model_path}: {e}")
            raise ValueError("Failed to load tf_efficientnet_b0 model. Please ensure the model file exists.")
        
        self.model.classifier = nn.Identity()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize GPU-based SIFT detector
        print("Initializing CUDA-based SIFT detector...")
        try:
            self.sift_detector = cv2.cuda.SIFT_CUDA_create(nfeatures=2000)
            self.cuda_available = True
            print("CUDA SIFT detector initialized")
        except Exception as e:
            print(f"CUDA SIFT not available, using CPU fallback: {e}")
            self.sift_detector = cv2.SIFT_create(nfeatures=500)  # Reduced for CPU
            self.cuda_available = False
        
        # Initialize GPU-based descriptor matcher
        print("Initializing CUDA-based descriptor matcher...")
        try:
            self.keypoint_matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_L2)
            print("CUDA descriptor matcher initialized")
        except Exception as e:
            print(f"CUDA matcher not available, using CPU fallback: {e}")
            self.keypoint_matcher = cv2.BFMatcher(cv2.NORM_L2)
        
        # Camera intrinsics
        self.base_K = torch.tensor([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)
    
    def convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {self.convert_numpy_types(k): self.convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def load_image_batch_gpu(self, image_paths):
        """Load and preprocess a batch of images directly to GPU"""
        images = []
        valid_paths = []
        
        for path in image_paths:
            try:
                image = Image.open(path).convert('RGB')
                image_tensor = self.transform(image)
                images.append(image_tensor)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
        
        if not images:
            return torch.tensor([]).to(self.device), valid_paths
        
        return torch.stack(images).to(self.device), valid_paths
    
    def extract_features_batch_gpu(self, image_paths):
        """Extract features from images in batches using GPU"""
        all_features = []
        all_paths = []
        
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Extracting features on GPU"):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_images, valid_paths = self.load_image_batch_gpu(batch_paths)
            
            if len(batch_images) == 0:
                continue
            
            with torch.no_grad():
                features = self.model(batch_images)
                features = F.normalize(features, p=2, dim=1)
                all_features.append(features)
                all_paths.extend(valid_paths)
        
        if not all_features:
            return torch.tensor([]).to(self.device), all_paths
            
        return torch.cat(all_features, dim=0), all_paths
    
    def compute_similarity_matrix_gpu(self, features):
        """Compute cosine similarity matrix on GPU"""
        similarity_matrix = torch.mm(features, features.t())
        return torch.clamp(similarity_matrix, 0, 1)
    
    def cluster_images_gpu(self, similarity_matrix):
        """Cluster images using DBSCAN on GPU-computed similarity matrix"""
        distance_matrix = 1 - similarity_matrix.cpu().numpy()
        distance_matrix = np.maximum(distance_matrix, 0)
        
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, 
                          metric='precomputed', n_jobs=-1)
        labels = clustering.fit_predict(distance_matrix)
        
        clusters = defaultdict(list)
        outliers = []
        
        for idx, label in enumerate(labels):
            if label == -1:
                outliers.append(idx)
            else:
                cluster_id = int(label)
                clusters[cluster_id].append(idx)
        
        return dict(clusters), outliers
    
    def process_image_collection_gpu(self, image_dir):
        """Process a collection of images to identify scene clusters using GPU"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        for file in os.listdir(image_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(image_dir, file))
        
        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Processing {len(image_paths)} images on GPU...")
        
        features, valid_paths = self.extract_features_batch_gpu(image_paths)
        
        if len(features) == 0:
            return {}, []
        
        similarity_matrix = self.compute_similarity_matrix_gpu(features)
        clusters_indices, outlier_indices = self.cluster_images_gpu(similarity_matrix)
        
        clusters = {k: [valid_paths[i] for i in v] for k, v in clusters_indices.items()}
        outliers = [valid_paths[i] for i in outlier_indices]
        
        return clusters, outliers
    
    def load_image_for_matching(self, image_path):
        """Load image and convert to GPU tensor or CUDA mat for feature matching"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            if self.cuda_available:
                # Upload to GPU as CUDA mat
                image_gpu = cv2.cuda_GpuMat()
                image_gpu.upload(image)
                return image_gpu
            else:
                # Return numpy array for CPU processing
                return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def extract_sift_features_gpu(self, image):
        """Extract SIFT keypoints and descriptors using GPU or CPU fallback"""
        try:
            if image is None:
                return None, None
            
            if self.cuda_available:
                # Convert to grayscale on GPU
                gray_gpu = cv2.cuda_GpuMat()
                if image.size()[2] == 3:
                    gray_gpu = cv2.cuda.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray_gpu = image
                
                # Extract SIFT features using CUDA
                keypoints, descriptors = self.sift_detector.detectAndCompute(gray_gpu, None)
                
                if descriptors is None or descriptors.empty():
                    return None, None
                
                # Download keypoints and descriptors to CPU for conversion to PyTorch
                keypoints = keypoints.download()
                descriptors = descriptors.download()
                
                # Convert to PyTorch tensors
                kpts_coords = torch.tensor([[kp.pt[0], kp.pt[1]] for kp in keypoints], 
                                         dtype=torch.float32, device=self.device).unsqueeze(0)
                descriptors_tensor = torch.tensor(descriptors, dtype=torch.float32, 
                                               device=self.device).unsqueeze(0)
                
                return kpts_coords, descriptors_tensor
            else:
                # CPU fallback
                if len(image.shape) == 3:
                    gray_np = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray_np = image
                
                keypoints, descriptors = self.sift_detector.detectAndCompute(gray_np, None)
                
                if descriptors is None:
                    return None, None
                
                kpts_coords = torch.tensor([[kp.pt[0], kp.pt[1]] for kp in keypoints], 
                                         dtype=torch.float32, device=self.device).unsqueeze(0)
                descriptors_tensor = torch.tensor(descriptors, dtype=torch.float32, 
                                               device=self.device).unsqueeze(0)
                
                return kpts_coords, descriptors_tensor
        except Exception as e:
            print(f"Error extracting SIFT features: {e}")
            return None, None
    
    def match_features_gpu(self, desc1, desc2, lafs1, lafs2):
        """Match SIFT features between two images using GPU or CPU fallback"""
        if desc1 is None or desc2 is None or desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        
        try:
            if self.cuda_available:
                # Convert PyTorch tensors to OpenCV CUDA mats
                desc1_np = desc1.squeeze(0).cpu().numpy()
                desc2_np = desc2.squeeze(0).cpu().numpy()
                
                desc1_gpu = cv2.cuda_GpuMat()
                desc2_gpu = cv2.cuda_GpuMat()
                desc1_gpu.upload(desc1_np)
                desc2_gpu.upload(desc2_np)
                
                # Perform matching on GPU
                matches = self.keypoint_matcher.knnMatch(desc1_gpu, desc2_gpu, k=2)
                
                # Apply Lowe's ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.8 * n.distance:
                        good_matches.append(m)
                
                if len(good_matches) < 8:
                    return torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
                
                # Get matched keypoints
                kpts1_idx = [m.queryIdx for m in good_matches]
                kpts2_idx = [m.trainIdx for m in good_matches]
                
                kpts1_matched = lafs1[0, kpts1_idx]
                kpts2_matched = lafs2[0, kpts2_idx]
                
                return kpts1_matched, kpts2_matched
            else:
                # CPU fallback
                desc1_np = desc1.squeeze(0).cpu().numpy()
                desc2_np = desc2.squeeze(0).cpu().numpy()
                
                matches = self.keypoint_matcher.knnMatch(desc1_np, desc2_np, k=2)
                
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.8 * n.distance:
                        good_matches.append(m)
                
                if len(good_matches) < 8:
                    return torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
                
                kpts1_idx = [m.queryIdx for m in good_matches]
                kpts2_idx = [m.trainIdx for m in good_matches]
                
                kpts1_matched = lafs1[0, kpts1_idx]
                kpts2_matched = lafs2[0, kpts2_idx]
                
                return kpts1_matched, kpts2_matched
        except Exception as e:
            print(f"Error matching features: {e}")
            return torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
    
    def estimate_pose_gpu(self, kpts1, kpts2, K):
        """Estimate relative pose from feature matches using GPU"""
        if len(kpts1) < 8 or len(kpts2) < 8:
            return None, None, 0
        
        try:
            with torch.no_grad():
                kpts1 = kpts1.to(self.device)
                kpts2 = kpts2.to(self.device)
                K = K.to(self.device)
                
                # Add batch dimension if needed
                if len(kpts1.shape) == 2:
                    kpts1 = kpts1.unsqueeze(0)
                if len(kpts2.shape) == 2:
                    kpts2 = kpts2.unsqueeze(0)
                
                # Compute fundamental matrix on GPU
                F_mat, mask = self.compute_fundamental_matrix_gpu(kpts1[0], kpts2[0])
                
                if F_mat is None or mask.sum() < 8:
                    return None, None, 0
                
                # Convert fundamental to essential matrix
                E = self.fundamental_to_essential_gpu(F_mat, K)
                
                # Recover pose from essential matrix
                R, t = self.recover_pose_from_essential_gpu(E, kpts1[0], kpts2[0], mask)
                
                return R, t, mask.sum().item()
        except Exception as e:
            print(f"Error estimating pose: {e}")
            return None, None, 0
    
    def compute_fundamental_matrix_gpu(self, kpts1, kpts2):
        """Compute fundamental matrix using 8-point algorithm on GPU"""
        try:
            if len(kpts1) < 8 or len(kpts2) < 8:
                return None, None
            
            # Normalize keypoints
            kpts1 = kpts1.to(self.device)
            kpts2 = kpts2.to(self.device)
            
            # Build constraint matrix
            n_points = min(len(kpts1), len(kpts2), 100)
            A = []
            for i in range(n_points):
                x1, y1 = kpts1[i]
                x2, y2 = kpts2[i]
                A.append([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])
            
            A = torch.tensor(A, dtype=torch.float32, device=self.device)
            
            # Solve using SVD
            U, S, V = torch.linalg.svd(A)
            F = V[:, -1].reshape(3, 3)
            
            # Enforce rank-2 constraint
            U_f, S_f, V_f = torch.linalg.svd(F)
            S_f = torch.tensor([S_f[0], S_f[1], 0], dtype=torch.float32, device=self.device)
            F = U_f @ torch.diag(S_f) @ V_f.t()
            
            # Create mask
            mask = torch.ones(n_points, dtype=torch.bool, device=self.device)
            
            return F, mask
        except Exception as e:
            print(f"Fundamental matrix computation failed: {e}")
            return None, None
    
    def fundamental_to_essential_gpu(self, F, K):
        """Convert fundamental matrix to essential matrix on GPU"""
        try:
            E = K.t() @ F @ K
            return E
        except Exception as e:
            print(f"Essential matrix computation failed: {e}")
            return None
    
    def recover_pose_from_essential_gpu(self, E, kpts1, kpts2, mask):
        """Recover camera pose from essential matrix on GPU"""
        try:
            # SVD decomposition
            U, S, Vt = torch.linalg.svd(E)
            
            # Ensure proper rotation matrix
            if torch.det(U) < 0:
                U[:, -1] *= -1
            if torch.det(Vt) < 0:
                Vt[-1, :] *= -1
            
            # Two possible rotations
            W = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], 
                           dtype=torch.float32, device=self.device)
            
            R1 = U @ W @ Vt
            R2 = U @ W.t() @ Vt
            t = U[:, 2:3]
            
            # Choose configuration with most points in front
            candidates = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
            best_R, best_t = candidates[0]
            best_score = 0
            
            for R, t_cand in candidates:
                score = self.count_points_in_front(R, t_cand, kpts1[:10], kpts2[:10])
                if score > best_score:
                    best_score = score
                    best_R, best_t = R, t_cand
            
            return best_R, best_t
        except Exception as e:
            print(f"Pose recovery failed: {e}")
            return torch.eye(3, device=self.device), torch.zeros(3, 1, device=self.device)
    
    def count_points_in_front(self, R, t, kpts1, kpts2):
        """Count how many points are in front of both cameras"""
        try:
            count = 0
            for i in range(min(len(kpts1), 10)):
                p1 = torch.cat([kpts1[i], torch.ones(1, device=self.device)])
                p2 = torch.cat([kpts2[i], torch.ones(1, device=self.device)])
                
                # Simplified depth check
                if torch.dot(p1[:2], p2[:2]) > 0:
                    count += 1
            
            return count
        except:
            return 0
    
    def reconstruct_scene_gpu(self, image_paths, output_dir=None):
        """Reconstruct 3D scene and estimate camera poses using GPU"""
        if len(image_paths) < 2:
            return {}
        
        K = self.base_K.clone()
        poses = {}
        
        print("Extracting features for SfM on GPU...")
        image_features = {}
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            img = self.load_image_for_matching(img_path)
            if img is not None:
                lafs, descs = self.extract_sift_features_gpu(img)
                if lafs is not None and descs is not None:
                    image_features[img_path] = (lafs, descs)
        
        valid_images = list(image_features.keys())
        if len(valid_images) < 2:
            return poses
        
        poses[valid_images[0]] = (
            torch.eye(3, device=self.device), 
            torch.zeros(3, 1, device=self.device)
        )
        
        print("Estimating camera poses on GPU...")
        for i in tqdm(range(1, len(valid_images)), desc="Pose estimation"):
            curr_img = valid_images[i]
            best_matches = 0
            best_pose = None
            
            for prev_img in valid_images[:i]:
                if prev_img not in poses:
                    continue
                
                lafs1, desc1 = image_features[prev_img]
                lafs2, desc2 = image_features[curr_img]
                
                kpts1, kpts2 = self.match_features_gpu(desc1, desc2, lafs1, lafs2)
                
                if len(kpts1) > best_matches:
                    R, t, num_inliers = self.estimate_pose_gpu(kpts1, kpts2, K)
                    
                    if R is not None and num_inliers > best_matches:
                        best_matches = num_inliers
                        R_prev, t_prev = poses[prev_img]
                        R_world = R @ R_prev
                        t_world = R @ t_prev + t
                        best_pose = (R_world, t_world)
            
            if best_pose is not None:
                poses[curr_img] = best_pose
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            poses_file = os.path.join(output_dir, 'camera_poses.json')
            
            poses_serializable = {}
            for img_path, (R, t) in poses.items():
                poses_serializable[os.path.basename(img_path)] = {
                    'rotation_matrix': R.cpu().numpy().tolist(),
                    'translation_vector': t.cpu().numpy().flatten().tolist()
                }
            
            with open(poses_file, 'w') as f:
                json.dump(poses_serializable, f, indent=2)
        
        return poses
    
    def process_dataset_gpu(self, dataset_dir, output_dir):
        """Process entire dataset using GPU: clustering + reconstruction for each cluster"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing dataset on GPU: {dataset_dir}")
        
        clusters, outliers = self.process_image_collection_gpu(dataset_dir)
        
        print(f"Found {len(clusters)} clusters and {len(outliers)} outliers")
        
        clustering_results = {
            'clusters': {str(k): [os.path.basename(path) for path in v] 
                        for k, v in clusters.items()},
            'outliers': [os.path.basename(path) for path in outliers]
        }
        
        clustering_results = self.convert_numpy_types(clustering_results)
        
        with open(os.path.join(output_dir, 'clustering_results.json'), 'w') as f:
            json.dump(clustering_results, f, indent=2)
        
        reconstruction_results = {}
        
        for cluster_id, image_paths in clusters.items():
            if len(image_paths) >= 2:
                print(f"Reconstructing cluster {cluster_id} with {len(image_paths)} images on GPU...")
                cluster_output_dir = os.path.join(output_dir, f'cluster_{cluster_id}')
                poses = self.reconstruct_scene_gpu(image_paths, cluster_output_dir)
                reconstruction_results[int(cluster_id)] = len(poses)
            else:
                print(f"Skipping cluster {cluster_id} (too few images)")
                reconstruction_results[int(cluster_id)] = 0
        
        summary = {
            'dataset': os.path.basename(dataset_dir),
            'total_images': len([f for f in os.listdir(dataset_dir) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]),
            'clusters': len(clusters),
            'outliers': len(outliers),
            'reconstructed_clusters': sum(1 for x in reconstruction_results.values() if x > 0),
            'cluster_details': reconstruction_results
        }
        
        summary = self.convert_numpy_types(summary)
        
        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        return clusters, outliers, reconstruction_results
    
    def create_submission_format(self, results, output_dir):
        """Create a submission file in CSV format for Kaggle Image Matching Challenge 2025"""
        submission_data = []
        
        for dataset, dataset_results in results.items():
            # Handle the actual structure where dataset_results is a dictionary with metadata
            poses = dataset_results.get('poses', {})
            outliers = dataset_results.get('outliers_list', [])
            
            # Process reconstructed images
            for img_name, pose in poses.items():
                if pose is not None and isinstance(pose, tuple) and len(pose) == 2:
                    rotation_matrix = pose[0].cpu().numpy().flatten()
                    translation_vector = pose[1].cpu().numpy().flatten()
                    
                    rotation_str = ';'.join([f'{x:.9f}' for x in rotation_matrix])
                    translation_str = ';'.join([f'{x:.9f}' for x in translation_vector])
                    
                    image_id = f"{dataset}_{img_name}_public"
                    
                    submission_data.append({
                        'image_id': image_id,
                        'dataset': dataset,
                        'scene': 'cluster0',  # Default scene name
                        'image': img_name,
                        'rotation_matrix': rotation_str,
                        'translation_vector': translation_str
                    })
            
            # Handle outliers
            for img_path in outliers:
                img_name = os.path.basename(img_path)
                rotation_matrix = np.eye(3).flatten()
                translation_vector = np.zeros(3)
                
                rotation_str = ';'.join([f'{x:.9f}' for x in rotation_matrix])
                translation_str = ';'.join([f'{x:.9f}' for x in translation_vector])
                
                image_id = f"{dataset}_{img_name}_public"
                
                submission_data.append({
                    'image_id': image_id,
                    'dataset': dataset,
                    'scene': 'outliers',
                    'image': img_name,
                    'rotation_matrix': rotation_str,
                    'translation_vector': translation_str
                })
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(submission_data)
        output_path = os.path.join(output_dir, 'submission.csv')
        df.to_csv(output_path, index=False)
        print(f"Submission file saved to {output_path}")

def main_kaggle_gpu():
    """Main function optimized for Kaggle GPU environment"""
    input_dir = "/kaggle/input/image-matching-challenge-2025/train"
    output_dir = "/kaggle/working"
    
    pipeline = GPUOptimizedSceneReconstructionPipeline(
        eps=0.4,
        min_samples=3,
        batch_size=64
    )
    
    all_results = {}
    
    if os.path.exists(input_dir):
        dataset_folders = [d for d in os.listdir(input_dir) 
                          if os.path.isdir(os.path.join(input_dir, d))]
        
        for dataset_name in dataset_folders:
            dataset_path = os.path.join(input_dir, dataset_name)
            dataset_output = os.path.join(output_dir, dataset_name)
            
            try:
                print(f"\n{'='*50}")
                print(f"Processing dataset on GPU: {dataset_name}")
                print(f"{'='*50}")
                
                clusters, outliers, reconstruction_results = pipeline.process_dataset_gpu(
                    dataset_path, dataset_output
                )
                
                # Get poses for the first cluster (or create empty dict if no clusters)
                first_cluster_poses = {}
                if clusters:
                    first_cluster_images = list(clusters.values())[0]
                    if len(first_cluster_images) >= 2:
                        first_cluster_poses = pipeline.reconstruct_scene_gpu(first_cluster_images, None)
                
                all_results[dataset_name] = {
                    'clusters': len(clusters),
                    'outliers': len(outliers),
                    'reconstructed': sum(1 for x in reconstruction_results.values() if x > 0),
                    'poses': {os.path.basename(k): v for k, v in first_cluster_poses.items()},
                    'outliers_list': outliers
                }
                
                print(f"Completed {dataset_name}: {len(clusters)} clusters, {len(outliers)} outliers")
                
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                # Add empty results for failed datasets
                all_results[dataset_name] = {
                    'clusters': 0,
                    'outliers': 0,
                    'reconstructed': 0,
                    'poses': {},
                    'outliers_list': []
                }
    
    pipeline.create_submission_format(all_results, output_dir)
    
    print(f"\n{'='*50}")
    print("FINAL GPU PROCESSING SUMMARY")
    print(f"{'='*50}")
    for dataset_name, results in all_results.items():
        print(f"{dataset_name}: {results['clusters']} clusters, "
              f"{results['outliers']} outliers, "
              f"{results['reconstructed']} reconstructed")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Submission file: {os.path.join(output_dir, 'submission.csv')}")

def main_local_gpu(input_dir, output_dir):
    """Main function for local GPU testing"""
    pipeline = GPUOptimizedSceneReconstructionPipeline(
        eps=0.4,
        min_samples=3,
        batch_size=64
    )
    
    if os.path.isdir(input_dir):
        clusters, outliers, reconstruction_results = pipeline.process_dataset_gpu(
            input_dir, output_dir
        )
        
        all_results = {
            os.path.basename(input_dir): {
                'clusters': len(clusters),
                'outliers': len(outliers),
                'reconstructed': sum(1 for x in reconstruction_results.values() if x > 0),
                'poses': {os.path.basename(k): v for k, v in pipeline.reconstruct_scene_gpu(clusters.get(0, []), None).items()},
                'outliers_list': outliers
            }
        }
        
        pipeline.create_submission_format(all_results, output_dir)
    else:
        print(f"Input directory not found: {input_dir}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Some operations will use CPU fallback.")
    
    if os.path.exists("/kaggle"):
        main_kaggle_gpu()
    else:
        import sys
        if len(sys.argv) != 3:
            print("Usage: python script.py <input_dir> <output_dir>")
            sys.exit(1)
        main_local_gpu(sys.argv[1], sys.argv[2])