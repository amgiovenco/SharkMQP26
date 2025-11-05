"""
Species-Specific Synthetic Data Generation with Comparison Tools
Generate and validate synthetic data for individual shark species
"""

import pandas as pd
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class SpeciesSpecificSynthesizer:
    def __init__(self, data, species_name, n_components=50):
        """
        Initialize synthesizer for a specific species
        
        Args:
            data: Full DataFrame with all species
            species_name: Name of species to synthesize
            n_components: number of PCA components (will auto-adjust if too large)
        """
        self.species_column = data.columns[0]
        self.all_species_data = data
        
        # Filter to specific species
        self.species_name = species_name
        self.species_data = data[data[self.species_column] == species_name].copy()
        
        if len(self.species_data) == 0:
            raise ValueError(f"No data found for species: {species_name}")
        
        self.temp_columns = list(data.columns[1:])
        
        # Auto-adjust n_components if larger than n_samples
        max_components = min(len(self.species_data) - 1, len(self.temp_columns))
        if n_components > max_components:
            print(f"Warning: Requested {n_components} components but only {len(self.species_data)} samples available")
            print(f"Auto-adjusting to {max_components} components")
            n_components = max_components
        
        self.n_components = n_components
        
        print(f"="*60)
        print(f"Species-Specific Synthesizer: {species_name}")
        print(f"="*60)
        print(f"Original samples for {species_name}: {len(self.species_data)}")
        print(f"Temperature columns: {len(self.temp_columns)}")
        print(f"PCA components: {n_components}")
        
    def fit(self):
        """Fit the synthesizer"""
        print(f"\n{'='*60}")
        print("Step 1: Applying PCA")
        print(f"{'='*60}")
        
        # Extract temperature data only (species is constant)
        temp_data = self.species_data[self.temp_columns].values
        
        # Apply PCA
        self.pca = PCA(n_components=self.n_components, random_state=8)
        pca_components = self.pca.fit_transform(temp_data)
        
        variance_explained = self.pca.explained_variance_ratio_.sum() * 100
        print(f"Variance explained: {variance_explained:.2f}%")
        
        # Create reduced dataframe (no species column needed since all same)
        pca_columns = [f'PC{i+1}' for i in range(self.n_components)]
        self.reduced_data = pd.DataFrame(pca_components, columns=pca_columns)
        
        # Setup metadata
        print(f"\n{'='*60}")
        print("Step 2: Setting up metadata")
        print(f"{'='*60}")
        
        self.metadata = Metadata.detect_from_dataframe(
            data=self.reduced_data,
            table_name='species_curves'
        )
        
        for col in pca_columns:
            self.metadata.update_column(
                column_name=col,
                sdtype='numerical',
                table_name='species_curves'
            )
        
        print(f"Metadata configured")
        
        # Fit synthesizer
        print(f"\n{'='*60}")
        print("Step 3: Training synthesizer")
        print(f"{'='*60}")
        
        self.synthesizer = GaussianCopulaSynthesizer(
            self.metadata,
            enforce_min_max_values=True
        )
        
        self.synthesizer.fit(self.reduced_data)
        print(f"Synthesizer trained!")
        
    def generate(self, num_rows):
        """Generate synthetic data"""
        print(f"\n{'='*60}")
        print(f"Generating {num_rows} synthetic samples")
        print(f"{'='*60}")
        
        # Generate in PCA space
        synthetic_pca = self.synthesizer.sample(num_rows=num_rows)
        
        # Reconstruct temperature curves
        synthetic_components = synthetic_pca.values
        reconstructed_temps = self.pca.inverse_transform(synthetic_components)
        
        # Create final dataframe with species column
        synthetic_data = pd.DataFrame(
            reconstructed_temps,
            columns=self.temp_columns
        )
        synthetic_data.insert(0, self.species_column, self.species_name)
        
        print(f"Generated {num_rows} synthetic samples for {self.species_name}")
        
        return synthetic_data
    
    def compare_distributions(self, synthetic_data, num_temps_to_plot=5):
        """
        Compare real vs synthetic data distributions
        
        Args:
            synthetic_data: Generated synthetic data
            num_temps_to_plot: Number of temperature points to visualize
        """
        print(f"\n{'='*60}")
        print("Statistical Comparison")
        print(f"{'='*60}")
        
        # Select evenly spaced temperature columns for comparison
        temp_indices = np.linspace(0, len(self.temp_columns)-1, num_temps_to_plot, dtype=int)
        selected_temps = [self.temp_columns[i] for i in temp_indices]
        
        print(f"\nComparing {num_temps_to_plot} temperature points:")
        print(f"{'Temperature':<15} {'Real Mean':<15} {'Syn Mean':<15} {'Real Std':<15} {'Syn Std':<15}")
        print("-" * 75)
        
        for temp in selected_temps:
            real_mean = self.species_data[temp].mean()
            syn_mean = synthetic_data[temp].mean()
            real_std = self.species_data[temp].std()
            syn_std = synthetic_data[temp].std()
            
            print(f"{temp:<15} {real_mean:<15.6f} {syn_mean:<15.6f} {real_std:<15.6f} {syn_std:<15.6f}")
    
    def plot_comparison(self, synthetic_data, num_samples=5, save_path=None):
        """
        Plot real vs synthetic melting curves
        
        Args:
            synthetic_data: Generated synthetic data
            num_samples: Number of curves to plot from each
            save_path: Optional path to save figure
        """
        print(f"\n{'='*60}")
        print("Plotting melting curves")
        print(f"{'='*60}")
        
        # Get temperature values
        try:
            temperatures = [float(col) for col in self.temp_columns]
        except:
            temperatures = list(range(len(self.temp_columns)))
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot real data
        real_samples = self.species_data.sample(min(num_samples, len(self.species_data)))
        for idx, (_, row) in enumerate(real_samples.iterrows()):
            fluorescence = row[self.temp_columns].values
            axes[0].plot(temperatures, fluorescence, alpha=0.7, label=f'Sample {idx+1}')
        
        axes[0].set_title(f'Real {self.species_name} Curves (n={len(self.species_data)})', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Temperature (°C)')
        axes[0].set_ylabel('Fluorescence')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot synthetic data
        syn_samples = synthetic_data.sample(min(num_samples, len(synthetic_data)))
        for idx, (_, row) in enumerate(syn_samples.iterrows()):
            fluorescence = row[self.temp_columns].values
            axes[1].plot(temperatures, fluorescence, alpha=0.7, label=f'Synthetic {idx+1}')
        
        axes[1].set_title(f'Synthetic {self.species_name} Curves (n={len(synthetic_data)})', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Temperature (°C)')
        axes[1].set_ylabel('Fluorescence')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
        print("Plot displayed")
    
    def plot_mean_curves(self, synthetic_data, save_path=None):
        """
        Plot mean curves with confidence intervals
        
        Args:
            synthetic_data: Generated synthetic data
            save_path: Optional path to save figure
        """
        try:
            temperatures = [float(col) for col in self.temp_columns]
        except:
            temperatures = list(range(len(self.temp_columns)))
        
        # Calculate means and stds
        real_mean = self.species_data[self.temp_columns].mean(axis=0).values
        real_std = self.species_data[self.temp_columns].std(axis=0).values
        syn_mean = synthetic_data[self.temp_columns].mean(axis=0).values
        syn_std = synthetic_data[self.temp_columns].std(axis=0).values
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot real data
        ax.plot(temperatures, real_mean, 'b-', linewidth=2, label='Real Mean', alpha=0.8)
        ax.fill_between(temperatures, real_mean - real_std, real_mean + real_std, 
                        alpha=0.2, color='blue', label='Real ±1 SD')
        
        # Plot synthetic data
        ax.plot(temperatures, syn_mean, 'r--', linewidth=2, label='Synthetic Mean', alpha=0.8)
        ax.fill_between(temperatures, syn_mean - syn_std, syn_mean + syn_std, 
                        alpha=0.2, color='red', label='Synthetic ±1 SD')
        
        ax.set_title(f'{self.species_name}: Real vs Synthetic Mean Curves', fontsize=14, fontweight='bold')
        ax.set_xlabel('Temperature (C)', fontsize=12)
        ax.set_ylabel('Fluorescence', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Mean curve figure saved to: {save_path}")
        
        plt.show()


def main():
    """Main execution with species-specific generation"""
    
    # Load data
    print("\nLoading shark melting curve data...")
    df = pd.read_csv('shark_dataset.csv')
    print(f"Loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    
    # Show available species
    species_column = df.columns[0]
    print(f"\nAvailable species ({df[species_column].nunique()} total):")
    species_counts = df[species_column].value_counts()
    for i, (species, count) in enumerate(species_counts.head(20).items()):
        print(f"  {i+1}. {species}: {count} samples")
    if len(species_counts) > 20:
        print(f"  ... and {len(species_counts) - 20} more species")
    
    # ==============================================
    # Choose species to generate, change as needed.
    # ==============================================
    target_species = "Atlantic Sharpnose shark"
    print(f"\nNOTE: For best results, choose a species with at least 10 samples")
    # ==============================================
    
    print(f"\n{'='*60}")
    print(f"Generating synthetic data for: {target_species}")
    print(f"{'='*60}")
    
    # Initialize species-specific synthesizer
    synth = SpeciesSpecificSynthesizer(
        data=df,
        species_name=target_species,
        n_components=50
    )
    
    # Fit the model
    synth.fit()
    
    # Generate synthetic samples
    num_synthetic = 10  # Generate same number as real samples
    synthetic_data = synth.generate(num_rows=num_synthetic)
    
    # Compare distributions
    synth.compare_distributions(synthetic_data, num_temps_to_plot=5)
    
    # Plot comparisons
    synth.plot_comparison(synthetic_data, num_samples=5, 
                          save_path=f'comparison_{target_species.replace(" ", "_")}.png')
    
    synth.plot_mean_curves(synthetic_data, 
                           save_path=f'mean_curves_{target_species.replace(" ", "_")}.png')
    
    # Save synthetic data
    output_file = f'synthetic_{target_species.replace(" ", "_")}.csv'
    synthetic_data.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"SUCCESS!")
    print(f"{'='*60}")
    print(f"Synthetic data saved to: {output_file}")
    print(f"Comparison plots saved")
    print(f"Real samples: {len(synth.species_data)}")
    print(f"Synthetic samples: {len(synthetic_data)}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()