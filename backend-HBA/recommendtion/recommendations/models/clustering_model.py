from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import joblib
import os

class UserClusteringModel:
    
    def __init__(self):
        self.user_clusterer = None
        self.room_clusterer = None
        self.scaler = StandardScaler()
        self.model_path = "./data/models/"
        
        # Ensure model directory exists
        os.makedirs(self.model_path, exist_ok=True)
    
    async def train_user_clusters(self, user_features: pd.DataFrame) -> Dict[str, Any]:
        """Train user clustering model"""
        
        # Prepare features
        feature_columns = [
            'avg_booking_duration', 'preferred_time_morning', 'preferred_time_afternoon',
            'booking_frequency', 'room_variety_score', 'advance_booking_days',
            'weekend_booking_ratio', 'cancellation_rate'
        ]
        
        X = user_features[feature_columns].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine optimal number of clusters
        optimal_k = self._find_optimal_clusters(X_scaled, max_k=10)
        
        # Train KMeans
        self.user_clusterer = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = self.user_clusterer.fit_predict(X_scaled)
        
        # Save models
        joblib.dump(self.user_clusterer, f"{self.model_path}/user_clusterer.joblib")
        joblib.dump(self.scaler, f"{self.model_path}/user_scaler.joblib")
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(user_features, cluster_labels)
        
        return {
            'n_clusters': optimal_k,
            'cluster_analysis': cluster_analysis,
            'silhouette_score': self._calculate_silhouette_score(X_scaled, cluster_labels)
        }
    
    async def predict_user_cluster(self, user_features: Dict[str, Any]) -> Tuple[int, float]:
        """Predict which cluster a user belongs to"""
        
        if self.user_clusterer is None:
            self._load_models()
        
        # Convert to feature vector
        feature_vector = self._user_dict_to_features(user_features)
        feature_vector_scaled = self.scaler.transform([feature_vector])
        
        # Predict cluster
        cluster = self.user_clusterer.predict(feature_vector_scaled)[0]
        
        # Calculate confidence (distance to cluster center)
        distances = self.user_clusterer.transform(feature_vector_scaled)[0]
        confidence = 1 / (1 + distances[cluster])  # Higher confidence for closer points
        
        return cluster, confidence
    
    async def get_cluster_recommendations(self, user_cluster: int) -> Dict[str, Any]:
        """Get recommendations based on cluster characteristics"""
        
        cluster_profiles = self._load_cluster_profiles()
        
        if user_cluster not in cluster_profiles:
            return {}
        
        profile = cluster_profiles[user_cluster]
        
        return {
            'recommended_rooms': profile.get('popular_rooms', []),
            'optimal_times': profile.get('preferred_times', []),
            'typical_duration': profile.get('avg_duration', 60),
            'booking_style': profile.get('booking_style', 'flexible'),
            'cluster_size': profile.get('cluster_size', 0)
        }