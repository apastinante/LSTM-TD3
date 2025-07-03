import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.stats import entropy
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class MemoryDebugger:
    """
    Advanced memory debugging and analysis toolkit for LSTM-TD3
    """
    
    def __init__(self, memory_dim, device='cpu'):
        self.memory_dim = memory_dim
        self.device = device
        self.memory_history = []
        self.context_history = []
        self.performance_history = []
        
    def log_memory_batch(self, extracted_memory, context_info, performance_metrics):
        """
        Log a batch of extracted memory states with context
        
        Args:
            extracted_memory: Tensor of shape (batch_size, memory_dim)
            context_info: Dict with context information (episode_step, reward, action, etc.)
            performance_metrics: Dict with performance metrics (q_values, loss, etc.)
        """
        memory_np = extracted_memory.detach().cpu().numpy()
        self.memory_history.append(memory_np)
        self.context_history.append(context_info)
        self.performance_history.append(performance_metrics)
        
    def analyze_memory_utilization(self, window_size=1000):
        """
        Analyze how well the memory dimensions are being utilized
        """
        if len(self.memory_history) < 10:
            print("Not enough memory history for analysis")
            return
            
        # Get recent memory states
        recent_memories = np.concatenate(self.memory_history[-window_size:], axis=0)
        
        # Analyze dimension-wise statistics
        dim_stats = {
            'mean': np.mean(recent_memories, axis=0),
            'std': np.std(recent_memories, axis=0),
            'min': np.min(recent_memories, axis=0),
            'max': np.max(recent_memories, axis=0),
            'range': np.max(recent_memories, axis=0) - np.min(recent_memories, axis=0)
        }
        
        # Identify dead/underutilized dimensions
        dead_dims = np.where(dim_stats['std'] < 1e-6)[0]
        low_util_dims = np.where((dim_stats['std'] < 0.1) & (dim_stats['std'] > 1e-6))[0]
        
        print(f"\n=== Memory Utilization Analysis ===")
        print(f"Total memory dimensions: {self.memory_dim}")
        print(f"Dead dimensions (std < 1e-6): {len(dead_dims)} - {dead_dims}")
        print(f"Low utilization dimensions (std < 0.1): {len(low_util_dims)} - {low_util_dims}")
        print(f"Well-utilized dimensions: {self.memory_dim - len(dead_dims) - len(low_util_dims)}")
        
        # Plot dimension utilization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Standard deviation by dimension
        axes[0,0].bar(range(self.memory_dim), dim_stats['std'])
        axes[0,0].set_title('Memory Dimension Standard Deviation')
        axes[0,0].set_xlabel('Dimension')
        axes[0,0].set_ylabel('Std Dev')
        axes[0,0].axhline(y=0.1, color='r', linestyle='--', label='Low utilization threshold')
        axes[0,0].legend()
        
        # Range by dimension
        axes[0,1].bar(range(self.memory_dim), dim_stats['range'])
        axes[0,1].set_title('Memory Dimension Range')
        axes[0,1].set_xlabel('Dimension')
        axes[0,1].set_ylabel('Range (Max - Min)')
        
        # Memory activation heatmap (sample)
        sample_memories = recent_memories[:min(100, len(recent_memories))]
        sns.heatmap(sample_memories.T, ax=axes[1,0], cmap='RdBu_r', center=0)
        axes[1,0].set_title('Memory Activation Heatmap (Sample)')
        axes[1,0].set_xlabel('Time Step')
        axes[1,0].set_ylabel('Memory Dimension')
        
        # Distribution of memory magnitudes
        memory_magnitudes = np.linalg.norm(recent_memories, axis=1)
        axes[1,1].hist(memory_magnitudes, bins=50, alpha=0.7)
        axes[1,1].set_title('Distribution of Memory Vector Magnitudes')
        axes[1,1].set_xlabel('L2 Norm')
        axes[1,1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        return dim_stats, dead_dims, low_util_dims
    
    def analyze_memory_dynamics(self, window_size=1000):
        """
        Analyze how memory states change over time
        """
        if len(self.memory_history) < 10:
            return
            
        recent_memories = np.concatenate(self.memory_history[-window_size:], axis=0)
        
        # Calculate memory state changes
        memory_changes = []
        for i in range(1, len(recent_memories)):
            change = np.linalg.norm(recent_memories[i] - recent_memories[i-1])
            memory_changes.append(change)
        
        # Calculate memory entropy (information content)
        memory_entropies = []
        for memory in recent_memories:
            # Normalize to probability distribution
            prob_dist = np.abs(memory) / (np.sum(np.abs(memory)) + 1e-8)
            mem_entropy = entropy(prob_dist + 1e-8)
            memory_entropies.append(mem_entropy)
        
        print(f"\n=== Memory Dynamics Analysis ===")
        print(f"Average memory change per step: {np.mean(memory_changes):.4f}")
        print(f"Memory change std: {np.std(memory_changes):.4f}")
        print(f"Average memory entropy: {np.mean(memory_entropies):.4f}")
        print(f"Memory entropy std: {np.std(memory_entropies):.4f}")
        
        # Plot dynamics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Memory changes over time
        axes[0,0].plot(memory_changes)
        axes[0,0].set_title('Memory State Changes Over Time')
        axes[0,0].set_xlabel('Time Step')
        axes[0,0].set_ylabel('L2 Norm of Change')
        
        # Memory entropy over time
        axes[0,1].plot(memory_entropies)
        axes[0,1].set_title('Memory Entropy Over Time')
        axes[0,1].set_xlabel('Time Step')
        axes[0,1].set_ylabel('Entropy')
        
        # Change distribution
        axes[1,0].hist(memory_changes, bins=50, alpha=0.7)
        axes[1,0].set_title('Distribution of Memory Changes')
        axes[1,0].set_xlabel('Change Magnitude')
        axes[1,0].set_ylabel('Frequency')
        
        # Entropy distribution
        axes[1,1].hist(memory_entropies, bins=50, alpha=0.7)
        axes[1,1].set_title('Distribution of Memory Entropy')
        axes[1,1].set_xlabel('Entropy')
        axes[1,1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        return memory_changes, memory_entropies
    
    def analyze_memory_performance_correlation(self, window_size=1000):
        """
        Analyze correlation between memory patterns and performance
        """
        if len(self.memory_history) < 10:
            return
            
        recent_memories = np.concatenate(self.memory_history[-window_size:], axis=0)
        recent_performance = self.performance_history[-len(recent_memories):]
        
        # Extract performance metrics
        q_values = [p.get('q_value', 0) for p in recent_performance]
        rewards = [p.get('reward', 0) for p in recent_performance]
        
        # Calculate correlations
        memory_magnitudes = np.linalg.norm(recent_memories, axis=1)
        
        # Correlation with Q-values
        if len(q_values) > 1:
            q_corr = np.corrcoef(memory_magnitudes[:len(q_values)], q_values)[0,1]
            print(f"\n=== Memory-Performance Correlation ===")
            print(f"Memory magnitude vs Q-value correlation: {q_corr:.4f}")
        
        # Correlation with rewards
        if len(rewards) > 1:
            reward_corr = np.corrcoef(memory_magnitudes[:len(rewards)], rewards)[0,1]
            print(f"Memory magnitude vs Reward correlation: {reward_corr:.4f}")
        
        # Dimensional analysis
        dim_q_corrs = []
        for dim in range(self.memory_dim):
            if len(q_values) > 1:
                corr = np.corrcoef(recent_memories[:len(q_values), dim], q_values)[0,1]
                dim_q_corrs.append(corr if not np.isnan(corr) else 0)
        
        # Plot correlations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Memory magnitude vs Q-values
        if len(q_values) > 1:
            axes[0,0].scatter(memory_magnitudes[:len(q_values)], q_values, alpha=0.6)
            axes[0,0].set_title(f'Memory Magnitude vs Q-Value (r={q_corr:.3f})')
            axes[0,0].set_xlabel('Memory Magnitude')
            axes[0,0].set_ylabel('Q-Value')
        
        # Memory magnitude vs Rewards
        if len(rewards) > 1:
            axes[0,1].scatter(memory_magnitudes[:len(rewards)], rewards, alpha=0.6)
            axes[0,1].set_title(f'Memory Magnitude vs Reward (r={reward_corr:.3f})')
            axes[0,1].set_xlabel('Memory Magnitude')
            axes[0,1].set_ylabel('Reward')
        
        # Dimensional correlations with Q-values
        if len(dim_q_corrs) > 0:
            axes[1,0].bar(range(len(dim_q_corrs)), dim_q_corrs)
            axes[1,0].set_title('Dimension-wise Q-Value Correlations')
            axes[1,0].set_xlabel('Memory Dimension')
            axes[1,0].set_ylabel('Correlation')
            axes[1,0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Memory evolution over performance
        performance_windows = np.array_split(recent_memories, 10)
        window_means = [np.mean(window, axis=0) for window in performance_windows]
        
        for i, mean_memory in enumerate(window_means):
            axes[1,1].plot(mean_memory, alpha=0.7, label=f'Window {i+1}')
        axes[1,1].set_title('Memory Evolution Across Performance Windows')
        axes[1,1].set_xlabel('Memory Dimension')
        axes[1,1].set_ylabel('Average Activation')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
        
        return dim_q_corrs
    
    def visualize_memory_clusters(self, window_size=1000, n_clusters=5):
        """
        Cluster memory states to identify distinct patterns
        """
        if len(self.memory_history) < 10:
            return
            
        recent_memories = np.concatenate(self.memory_history[-window_size:], axis=0)
        
        # Dimensionality reduction for visualization
        if self.memory_dim > 2:
            # PCA for linear projection
            pca = PCA(n_components=2)
            memory_pca = pca.fit_transform(recent_memories)
            
            # t-SNE for nonlinear projection
            if len(recent_memories) > 30:  # t-SNE requires enough samples
                tsne = TSNE(n_components=2, random_state=42)
                memory_tsne = tsne.fit_transform(recent_memories[:1000])  # Limit for t-SNE
            else:
                memory_tsne = memory_pca
        else:
            memory_pca = recent_memories
            memory_tsne = recent_memories
        
        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(recent_memories)
        
        print(f"\n=== Memory Clustering Analysis ===")
        print(f"Number of clusters: {n_clusters}")
        for i in range(n_clusters):
            cluster_size = np.sum(clusters == i)
            print(f"Cluster {i}: {cluster_size} samples ({cluster_size/len(clusters)*100:.1f}%)")
        
        # Plot clusters
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # PCA visualization
        scatter = axes[0,0].scatter(memory_pca[:, 0], memory_pca[:, 1], 
                                  c=clusters, cmap='tab10', alpha=0.6)
        axes[0,0].set_title('Memory States (PCA)')
        axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.colorbar(scatter, ax=axes[0,0])
        
        # t-SNE visualization
        if len(memory_tsne) > 0:
            scatter = axes[0,1].scatter(memory_tsne[:, 0], memory_tsne[:, 1], 
                                      c=clusters[:len(memory_tsne)], cmap='tab10', alpha=0.6)
            axes[0,1].set_title('Memory States (t-SNE)')
            axes[0,1].set_xlabel('t-SNE 1')
            axes[0,1].set_ylabel('t-SNE 2')
            plt.colorbar(scatter, ax=axes[0,1])
        
        # Cluster centroids
        centroids = kmeans.cluster_centers_
        axes[1,0].imshow(centroids.T, aspect='auto', cmap='RdBu_r')
        axes[1,0].set_title('Cluster Centroids')
        axes[1,0].set_xlabel('Cluster')
        axes[1,0].set_ylabel('Memory Dimension')
        
        # Cluster sizes
        cluster_sizes = [np.sum(clusters == i) for i in range(n_clusters)]
        axes[1,1].bar(range(n_clusters), cluster_sizes)
        axes[1,1].set_title('Cluster Sizes')
        axes[1,1].set_xlabel('Cluster')
        axes[1,1].set_ylabel('Number of Samples')
        
        plt.tight_layout()
        plt.show()
        
        return clusters, centroids
    
    def detect_memory_anomalies(self, window_size=1000, anomaly_threshold=2.0):
        """
        Detect anomalous memory states that might indicate learning issues
        """
        if len(self.memory_history) < 10:
            return
            
        recent_memories = np.concatenate(self.memory_history[-window_size:], axis=0)
        
        # Calculate memory magnitudes
        memory_magnitudes = np.linalg.norm(recent_memories, axis=1)
        
        # Detect anomalies using z-score
        mean_mag = np.mean(memory_magnitudes)
        std_mag = np.std(memory_magnitudes)
        z_scores = np.abs(memory_magnitudes - mean_mag) / (std_mag + 1e-8)
        
        anomaly_indices = np.where(z_scores > anomaly_threshold)[0]
        
        print(f"\n=== Memory Anomaly Detection ===")
        print(f"Anomaly threshold (z-score): {anomaly_threshold}")
        print(f"Number of anomalies detected: {len(anomaly_indices)}")
        print(f"Anomaly rate: {len(anomaly_indices)/len(recent_memories)*100:.2f}%")
        
        if len(anomaly_indices) > 0:
            print(f"Anomalous magnitudes: {memory_magnitudes[anomaly_indices]}")
            print(f"Normal magnitude range: [{mean_mag - 2*std_mag:.3f}, {mean_mag + 2*std_mag:.3f}]")
        
        # Plot anomalies
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Memory magnitudes over time
        axes[0,0].plot(memory_magnitudes, alpha=0.7)
        axes[0,0].scatter(anomaly_indices, memory_magnitudes[anomaly_indices], 
                         color='red', s=50, label='Anomalies')
        axes[0,0].axhline(y=mean_mag + anomaly_threshold*std_mag, 
                         color='r', linestyle='--', alpha=0.5)
        axes[0,0].axhline(y=mean_mag - anomaly_threshold*std_mag, 
                         color='r', linestyle='--', alpha=0.5)
        axes[0,0].set_title('Memory Magnitudes with Anomalies')
        axes[0,0].set_xlabel('Time Step')
        axes[0,0].set_ylabel('Memory Magnitude')
        axes[0,0].legend()
        
        # Z-score distribution
        axes[0,1].hist(z_scores, bins=50, alpha=0.7)
        axes[0,1].axvline(x=anomaly_threshold, color='r', linestyle='--', 
                         label=f'Threshold ({anomaly_threshold})')
        axes[0,1].set_title('Z-Score Distribution')
        axes[0,1].set_xlabel('Z-Score')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        
        # Anomalous memory patterns
        if len(anomaly_indices) > 0:
            anomalous_memories = recent_memories[anomaly_indices]
            normal_memories = recent_memories[z_scores <= anomaly_threshold]
            
            axes[1,0].plot(np.mean(normal_memories, axis=0), label='Normal Average', alpha=0.7)
            axes[1,0].plot(np.mean(anomalous_memories, axis=0), label='Anomalous Average', alpha=0.7)
            axes[1,0].set_title('Normal vs Anomalous Memory Patterns')
            axes[1,0].set_xlabel('Memory Dimension')
            axes[1,0].set_ylabel('Average Activation')
            axes[1,0].legend()
        
        # Anomaly timeline
        anomaly_timeline = np.zeros(len(recent_memories))
        anomaly_timeline[anomaly_indices] = 1
        axes[1,1].plot(anomaly_timeline)
        axes[1,1].set_title('Anomaly Timeline')
        axes[1,1].set_xlabel('Time Step')
        axes[1,1].set_ylabel('Anomaly (1=Yes, 0=No)')
        
        plt.tight_layout()
        plt.show()
        
        return anomaly_indices, z_scores
    
    def generate_memory_report(self, save_path=None):
        """
        Generate a comprehensive memory analysis report
        """
        print("\n" + "="*60)
        print("LSTM-TD3 MEMORY ANALYSIS REPORT")
        print("="*60)
        
        # Run all analyses
        dim_stats, dead_dims, low_util_dims = self.analyze_memory_utilization()
        memory_changes, memory_entropies = self.analyze_memory_dynamics()
        dim_q_corrs = self.analyze_memory_performance_correlation()
        clusters, centroids = self.visualize_memory_clusters()
        anomaly_indices, z_scores = self.detect_memory_anomalies()
        
        # Generate summary
        summary = {
            'total_samples': sum(len(m) for m in self.memory_history),
            'memory_dim': self.memory_dim,
            'dead_dimensions': len(dead_dims),
            'low_utilization_dimensions': len(low_util_dims),
            'avg_memory_change': np.mean(memory_changes),
            'avg_memory_entropy': np.mean(memory_entropies),
            'anomaly_rate': len(anomaly_indices) / sum(len(m) for m in self.memory_history) * 100,
            'top_correlated_dims': np.argsort(np.abs(dim_q_corrs))[-5:] if len(dim_q_corrs) > 0 else []
        }
        
        print(f"\nSUMMARY:")
        print(f"- Total samples analyzed: {summary['total_samples']}")
        print(f"- Memory utilization: {summary['memory_dim'] - summary['dead_dimensions'] - summary['low_utilization_dimensions']}/{summary['memory_dim']} dimensions well-utilized")
        print(f"- Memory stability: {summary['avg_memory_change']:.4f} average change per step")
        print(f"- Memory complexity: {summary['avg_memory_entropy']:.4f} average entropy")
        print(f"- Anomaly rate: {summary['anomaly_rate']:.2f}%")
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nReport saved to: {save_path}")
        
        return summary


def create_enhanced_memory_logging(original_compute_loss_q, original_compute_loss_pi, memory_debugger):
    """
    Enhance the original loss computation functions with detailed memory logging
    """
    
    def enhanced_compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        h_o, h_a, h_o2, h_a2, h_o_len, h_o2_len = data['hist_obs'], data['hist_act'], data['hist_obs2'], data['hist_act2'], data['hist_obs_len'], data['hist_obs2_len']

        # Get original results
        result = original_compute_loss_q(data)
        loss_q, loss_info = result
        
        # Enhanced memory logging
        q1_memory = loss_info['Q1ExtractedMemory']
        q2_memory = loss_info['Q2ExtractedMemory']
        
        # Log to memory debugger
        context_info = {
            'rewards': r.detach().cpu().numpy(),
            'episode_step': None,  # Could be passed in data
            'action_magnitude': torch.norm(a, dim=1).detach().cpu().numpy()
        }
        
        performance_metrics = {
            'q_value': loss_info['Q1Vals'],
            'reward': r.detach().cpu().numpy(),
            'loss': loss_q.item()
        }
        
        memory_debugger.log_memory_batch(
            torch.tensor(q1_memory), 
            context_info, 
            performance_metrics
        )
        
        return result
    
    def enhanced_compute_loss_pi(data):
        result = original_compute_loss_pi(data)
        loss_pi, loss_info = result
        
        # Log actor memory
        actor_memory = loss_info['ActExtractedMemory']
        
        context_info = {
            'episode_step': None,
            'policy_update': True
        }
        
        performance_metrics = {
            'policy_loss': loss_pi.item()
        }
        
        memory_debugger.log_memory_batch(
            torch.tensor(actor_memory),
            context_info,
            performance_metrics
        )
        
        return result
    
    return enhanced_compute_loss_q, enhanced_compute_loss_pi 