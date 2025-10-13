# Bioinformatics Pipeline Template

A production-ready bioinformatics pipeline template for genomic data analysis, featuring modern tools, machine learning, and cloud computing for 2025.

## 🚀 Features

- **Genomic Data Processing** - FASTQ, BAM, VCF file handling
- **Sequence Alignment** - BWA, Bowtie2, STAR integration
- **Variant Calling** - GATK, FreeBayes, DeepVariant
- **RNA-seq Analysis** - DESeq2, edgeR, limma-voom
- **Machine Learning** - Deep learning for genomics
- **Cloud Computing** - AWS, GCP, Azure integration
- **Containerization** - Docker, Singularity support
- **Workflow Management** - Nextflow, Snakemake, WDL
- **Data Visualization** - Interactive plots and dashboards
- **Reproducibility** - Conda, Bioconda environments
- **Quality Control** - FastQC, MultiQC, Picard
- **Database Integration** - Ensembl, NCBI, UniProt

## 📋 Prerequisites

- Python 3.9+
- R 4.2+
- Conda/Mamba
- Docker
- Git

## 🛠️ Quick Start

### 1. Create New Pipeline

```bash
git clone <this-repo> my-bioinformatics-pipeline
cd my-bioinformatics-pipeline
```

### 2. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate bioinformatics

# Or use mamba for faster installation
mamba env create -f environment.yml
mamba activate bioinformatics
```

### 3. Configure Pipeline

```bash
cp config/config.yaml.example config/config.yaml
# Edit configuration file
```

### 4. Run Pipeline

```bash
# Run with Nextflow
nextflow run main.nf -c config/config.yaml

# Run with Snakemake
snakemake --cores 4 --configfile config/config.yaml

# Run with Python
python src/main.py --config config/config.yaml
```

## 📁 Project Structure

```
├── src/                        # Source code
│   ├── pipelines/             # Pipeline implementations
│   │   ├── rnaseq/           # RNA-seq analysis pipeline
│   │   ├── variant_calling/  # Variant calling pipeline
│   │   ├── chip_seq/         # ChIP-seq analysis pipeline
│   │   └── metagenomics/     # Metagenomics pipeline
│   ├── tools/                # Custom tools and utilities
│   │   ├── sequence_analysis.py
│   │   ├── variant_analysis.py
│   │   ├── expression_analysis.py
│   │   └── visualization.py
│   ├── ml/                   # Machine learning modules
│   │   ├── deep_learning.py
│   │   ├── feature_extraction.py
│   │   └── model_training.py
│   └── utils/                # Utility functions
│       ├── file_handling.py
│       ├── database_utils.py
│       └── quality_control.py
├── workflows/                 # Workflow definitions
│   ├── nextflow/             # Nextflow workflows
│   ├── snakemake/            # Snakemake workflows
│   └── wdl/                  # WDL workflows
├── config/                   # Configuration files
│   ├── config.yaml
│   ├── samples.csv
│   └── reference_genomes.yaml
├── data/                     # Data directories
│   ├── raw/                  # Raw sequencing data
│   ├── processed/            # Processed data
│   ├── results/              # Analysis results
│   └── reference/            # Reference genomes
├── tests/                    # Test files
├── docs/                     # Documentation
└── environment.yml           # Conda environment
```

## 🔧 Available Scripts

```bash
# Pipeline Execution
nextflow run main.nf                    # Run main pipeline
snakemake --cores 4                     # Run with Snakemake
python src/main.py                      # Run Python pipeline

# Quality Control
python src/tools/quality_control.py     # Run QC analysis
fastqc data/raw/*.fastq.gz              # FastQC analysis
multiqc results/qc/                     # MultiQC report

# Data Processing
python src/tools/sequence_analysis.py   # Sequence analysis
python src/tools/variant_analysis.py    # Variant analysis
python src/tools/expression_analysis.py # Expression analysis

# Machine Learning
python src/ml/deep_learning.py          # Deep learning models
python src/ml/feature_extraction.py     # Feature extraction
python src/ml/model_training.py         # Model training

# Visualization
python src/tools/visualization.py       # Generate plots
jupyter notebook notebooks/             # Interactive analysis
```

## 🧬 RNA-seq Analysis Pipeline

### Nextflow Implementation

```groovy
// workflows/nextflow/rnaseq.nf
#!/usr/bin/env nextflow

params.reads = "data/raw/*_{1,2}.fastq.gz"
params.outdir = "results"
params.genome = "hg38"
params.annotation = "data/reference/annotation.gtf"

process QUALITY_CONTROL {
    publishDir "${params.outdir}/qc", mode: 'copy'
    
    input:
    tuple val(sample), path(reads)
    
    output:
    path("${sample}_fastqc.html"), emit: fastqc_html
    path("${sample}_fastqc.zip"), emit: fastqc_zip
    
    script:
    """
    fastqc ${reads} -o .
    """
}

process TRIMMING {
    publishDir "${params.outdir}/trimmed", mode: 'copy'
    
    input:
    tuple val(sample), path(reads)
    
    output:
    tuple val(sample), path("${sample}_trimmed_{1,2}.fastq.gz"), emit: trimmed_reads
    
    script:
    """
    trimmomatic PE ${reads} \\
        ${sample}_trimmed_1.fastq.gz ${sample}_trimmed_1_unpaired.fastq.gz \\
        ${sample}_trimmed_2.fastq.gz ${sample}_trimmed_2_unpaired.fastq.gz \\
        ILLUMINACLIP:TruSeq3-PE.fa:2:30:10 \\
        LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36
    """
}

process ALIGNMENT {
    publishDir "${params.outdir}/alignment", mode: 'copy'
    
    input:
    tuple val(sample), path(reads)
    
    output:
    path("${sample}.bam"), emit: bam
    path("${sample}.bam.bai"), emit: bai
    
    script:
    """
    STAR --genomeDir ${params.genome} \\
         --readFilesIn ${reads} \\
         --readFilesCommand zcat \\
         --outSAMtype BAM SortedByCoordinate \\
         --outFileNamePrefix ${sample}.
    """
}

process QUANTIFICATION {
    publishDir "${params.outdir}/quantification", mode: 'copy'
    
    input:
    path(bam)
    
    output:
    path("${bam.baseName}_counts.txt"), emit: counts
    
    script:
    """
    featureCounts -a ${params.annotation} \\
                  -o ${bam.baseName}_counts.txt \\
                  ${bam}
    """
}

process DIFFERENTIAL_EXPRESSION {
    publishDir "${params.outdir}/differential_expression", mode: 'copy'
    
    input:
    path(counts)
    
    output:
    path("deseq2_results.csv"), emit: results
    
    script:
    """
    Rscript scripts/deseq2_analysis.R ${counts}
    """
}

workflow {
    Channel.fromFilePairs(params.reads)
        .map { it -> [it[0], it[1]] }
        .set { reads_ch }
    
    reads_ch
        .into { qc_ch; trim_ch }
    
    qc_ch | QUALITY_CONTROL
    trim_ch | TRIMMING | ALIGNMENT | QUANTIFICATION | DIFFERENTIAL_EXPRESSION
}
```

### Python Implementation

```python
# src/pipelines/rnaseq/rnaseq_pipeline.py
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import subprocess
import logging

class RNASeqPipeline:
    """RNA-seq analysis pipeline."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.setup_directories()
    
    def setup_directories(self):
        """Create output directories."""
        dirs = [
            'results/qc',
            'results/trimmed',
            'results/alignment',
            'results/quantification',
            'results/differential_expression'
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def quality_control(self, fastq_files: List[str]) -> List[str]:
        """Run quality control analysis."""
        self.logger.info("Running quality control...")
        
        qc_files = []
        for fastq_file in fastq_files:
            sample_name = Path(fastq_file).stem
            output_dir = f"results/qc/{sample_name}"
            
            cmd = [
                'fastqc',
                fastq_file,
                '-o', output_dir
            ]
            
            subprocess.run(cmd, check=True)
            qc_files.append(f"{output_dir}/{sample_name}_fastqc.html")
        
        return qc_files
    
    def trim_reads(self, fastq_files: List[str]) -> List[str]:
        """Trim sequencing reads."""
        self.logger.info("Trimming reads...")
        
        trimmed_files = []
        for fastq_file in fastq_files:
            sample_name = Path(fastq_file).stem
            output_prefix = f"results/trimmed/{sample_name}_trimmed"
            
            cmd = [
                'trimmomatic',
                'PE',
                fastq_file,
                f"{output_prefix}_1.fastq.gz",
                f"{output_prefix}_1_unpaired.fastq.gz",
                f"{output_prefix}_2.fastq.gz",
                f"{output_prefix}_2_unpaired.fastq.gz",
                'ILLUMINACLIP:TruSeq3-PE.fa:2:30:10',
                'LEADING:3',
                'TRAILING:3',
                'SLIDINGWINDOW:4:15',
                'MINLEN:36'
            ]
            
            subprocess.run(cmd, check=True)
            trimmed_files.append(f"{output_prefix}_1.fastq.gz")
        
        return trimmed_files
    
    def align_reads(self, trimmed_files: List[str]) -> List[str]:
        """Align reads to reference genome."""
        self.logger.info("Aligning reads...")
        
        bam_files = []
        for trimmed_file in trimmed_files:
            sample_name = Path(trimmed_file).stem.replace('_trimmed_1', '')
            output_prefix = f"results/alignment/{sample_name}"
            
            cmd = [
                'STAR',
                '--genomeDir', self.config['genome_dir'],
                '--readFilesIn', trimmed_file,
                '--readFilesCommand', 'zcat',
                '--outSAMtype', 'BAM', 'SortedByCoordinate',
                '--outFileNamePrefix', output_prefix
            ]
            
            subprocess.run(cmd, check=True)
            bam_files.append(f"{output_prefix}Aligned.sortedByCoord.out.bam")
        
        return bam_files
    
    def quantify_expression(self, bam_files: List[str]) -> str:
        """Quantify gene expression."""
        self.logger.info("Quantifying expression...")
        
        counts_file = "results/quantification/counts.txt"
        
        cmd = [
            'featureCounts',
            '-a', self.config['annotation_file'],
            '-o', counts_file
        ] + bam_files
        
        subprocess.run(cmd, check=True)
        return counts_file
    
    def differential_expression(self, counts_file: str) -> str:
        """Perform differential expression analysis."""
        self.logger.info("Running differential expression analysis...")
        
        # Load count data
        counts_df = pd.read_csv(counts_file, sep='\t', index_col=0)
        
        # Remove metadata columns
        counts_df = counts_df.iloc[:, 5:]
        
        # Create sample metadata
        sample_metadata = self._create_sample_metadata(counts_df.columns)
        
        # Run DESeq2 analysis
        results_file = self._run_deseq2(counts_df, sample_metadata)
        
        return results_file
    
    def _create_sample_metadata(self, sample_names: List[str]) -> pd.DataFrame:
        """Create sample metadata DataFrame."""
        metadata = []
        for sample in sample_names:
            # Extract condition from sample name (assuming format: condition_replicate)
            condition = sample.split('_')[0]
            metadata.append({
                'sample': sample,
                'condition': condition
            })
        
        return pd.DataFrame(metadata)
    
    def _run_deseq2(self, counts_df: pd.DataFrame, metadata: pd.DataFrame) -> str:
        """Run DESeq2 analysis using R."""
        # Save data for R script
        counts_df.to_csv('temp_counts.csv')
        metadata.to_csv('temp_metadata.csv', index=False)
        
        # Run R script
        r_script = """
        library(DESeq2)
        library(dplyr)
        
        # Load data
        counts <- read.csv('temp_counts.csv', row.names=1)
        metadata <- read.csv('temp_metadata.csv')
        
        # Create DESeq2 object
        dds <- DESeqDataSetFromMatrix(countData=counts,
                                    colData=metadata,
                                    design=~condition)
        
        # Run DESeq2
        dds <- DESeq(dds)
        
        # Get results
        results <- results(dds)
        results <- as.data.frame(results)
        results <- results[order(results$padj),]
        
        # Save results
        write.csv(results, 'results/differential_expression/deseq2_results.csv')
        """
        
        with open('temp_deseq2.R', 'w') as f:
            f.write(r_script)
        
        subprocess.run(['Rscript', 'temp_deseq2.R'], check=True)
        
        # Clean up temporary files
        os.remove('temp_counts.csv')
        os.remove('temp_metadata.csv')
        os.remove('temp_deseq2.R')
        
        return 'results/differential_expression/deseq2_results.csv'
    
    def run_pipeline(self, fastq_files: List[str]) -> Dict[str, str]:
        """Run complete RNA-seq pipeline."""
        self.logger.info("Starting RNA-seq pipeline...")
        
        # Quality control
        qc_files = self.quality_control(fastq_files)
        
        # Trim reads
        trimmed_files = self.trim_reads(fastq_files)
        
        # Align reads
        bam_files = self.align_reads(trimmed_files)
        
        # Quantify expression
        counts_file = self.quantify_expression(bam_files)
        
        # Differential expression
        results_file = self.differential_expression(counts_file)
        
        return {
            'qc_files': qc_files,
            'trimmed_files': trimmed_files,
            'bam_files': bam_files,
            'counts_file': counts_file,
            'results_file': results_file
        }
```

## 🧬 Variant Calling Pipeline

```python
# src/pipelines/variant_calling/variant_pipeline.py
import os
import subprocess
from pathlib import Path
from typing import List, Dict
import logging

class VariantCallingPipeline:
    """Variant calling analysis pipeline."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def align_reads(self, fastq_files: List[str]) -> List[str]:
        """Align reads using BWA."""
        self.logger.info("Aligning reads with BWA...")
        
        bam_files = []
        for fastq_file in fastq_files:
            sample_name = Path(fastq_file).stem
            output_prefix = f"results/alignment/{sample_name}"
            
            # BWA alignment
            cmd = [
                'bwa', 'mem',
                '-t', '4',
                self.config['reference_genome'],
                fastq_file
            ]
            
            with open(f"{output_prefix}.sam", 'w') as f:
                subprocess.run(cmd, stdout=f, check=True)
            
            # Convert to BAM and sort
            cmd = [
                'samtools', 'view',
                '-bS', f"{output_prefix}.sam"
            ]
            
            with open(f"{output_prefix}.bam", 'w') as f:
                subprocess.run(cmd, stdout=f, check=True)
            
            # Sort BAM
            cmd = [
                'samtools', 'sort',
                f"{output_prefix}.bam",
                '-o', f"{output_prefix}_sorted.bam"
            ]
            
            subprocess.run(cmd, check=True)
            
            # Index BAM
            cmd = [
                'samtools', 'index',
                f"{output_prefix}_sorted.bam"
            ]
            
            subprocess.run(cmd, check=True)
            
            bam_files.append(f"{output_prefix}_sorted.bam")
            
            # Clean up SAM file
            os.remove(f"{output_prefix}.sam")
            os.remove(f"{output_prefix}.bam")
        
        return bam_files
    
    def mark_duplicates(self, bam_files: List[str]) -> List[str]:
        """Mark duplicate reads using Picard."""
        self.logger.info("Marking duplicates...")
        
        dedup_files = []
        for bam_file in bam_files:
            sample_name = Path(bam_file).stem.replace('_sorted', '')
            output_file = f"results/alignment/{sample_name}_dedup.bam"
            metrics_file = f"results/alignment/{sample_name}_metrics.txt"
            
            cmd = [
                'picard', 'MarkDuplicates',
                'I=', bam_file,
                'O=', output_file,
                'M=', metrics_file,
                'REMOVE_DUPLICATES=false'
            ]
            
            subprocess.run(cmd, check=True)
            
            # Index deduplicated BAM
            cmd = ['samtools', 'index', output_file]
            subprocess.run(cmd, check=True)
            
            dedup_files.append(output_file)
        
        return dedup_files
    
    def call_variants(self, bam_files: List[str]) -> str:
        """Call variants using GATK."""
        self.logger.info("Calling variants with GATK...")
        
        # Create GVCF files for each sample
        gvcf_files = []
        for bam_file in bam_files:
            sample_name = Path(bam_file).stem.replace('_dedup', '')
            gvcf_file = f"results/variants/{sample_name}.g.vcf"
            
            cmd = [
                'gatk', 'HaplotypeCaller',
                '-R', self.config['reference_genome'],
                '-I', bam_file,
                '-O', gvcf_file,
                '-ERC', 'GVCF'
            ]
            
            subprocess.run(cmd, check=True)
            gvcf_files.append(gvcf_file)
        
        # Combine GVCF files
        combined_gvcf = "results/variants/combined.g.vcf"
        cmd = [
            'gatk', 'CombineGVCFs',
            '-R', self.config['reference_genome'],
            '-O', combined_gvcf
        ] + [f'-V {gvcf}' for gvcf in gvcf_files]
        
        subprocess.run(cmd, check=True)
        
        # Genotype GVCF
        vcf_file = "results/variants/final.vcf"
        cmd = [
            'gatk', 'GenotypeGVCFs',
            '-R', self.config['reference_genome'],
            '-V', combined_gvcf,
            '-O', vcf_file
        ]
        
        subprocess.run(cmd, check=True)
        
        return vcf_file
    
    def filter_variants(self, vcf_file: str) -> str:
        """Filter variants using GATK."""
        self.logger.info("Filtering variants...")
        
        filtered_vcf = "results/variants/filtered.vcf"
        
        cmd = [
            'gatk', 'VariantFiltration',
            '-R', self.config['reference_genome'],
            '-V', vcf_file,
            '-O', filtered_vcf,
            '--filter-expression', 'QD < 2.0',
            '--filter-name', 'QD2',
            '--filter-expression', 'QUAL < 30.0',
            '--filter-name', 'QUAL30',
            '--filter-expression', 'SOR > 3.0',
            '--filter-name', 'SOR3',
            '--filter-expression', 'FS > 60.0',
            '--filter-name', 'FS60',
            '--filter-expression', 'MQ < 40.0',
            '--filter-name', 'MQ40'
        ]
        
        subprocess.run(cmd, check=True)
        
        return filtered_vcf
```

## 🤖 Machine Learning for Genomics

```python
# src/ml/deep_learning.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, List
import logging

class GenomicDataset(Dataset):
    """Dataset for genomic data."""
    
    def __init__(self, sequences: List[str], labels: List[int], max_length: int = 1000):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        self.vocab = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Convert sequence to tensor
        encoded = [self.vocab.get(base, 4) for base in sequence.upper()]
        
        # Pad or truncate to max_length
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        else:
            encoded.extend([4] * (self.max_length - len(encoded)))
        
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class GenomicCNN(nn.Module):
    """Convolutional Neural Network for genomic sequences."""
    
    def __init__(self, vocab_size: int = 5, embedding_dim: int = 128, 
                 num_classes: int = 2, max_length: int = 1000):
        super(GenomicCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        
        # Convolutional layers
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Classification
        x = self.classifier(x)
        
        return x

class GenomicTransformer(nn.Module):
    """Transformer model for genomic sequences."""
    
    def __init__(self, vocab_size: int = 5, d_model: int = 128, 
                 nhead: int = 8, num_layers: int = 6, 
                 num_classes: int = 2, max_length: int = 1000):
        super(GenomicTransformer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(max_length, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        seq_len = x.size(1)
        
        # Embedding
        x = self.embedding(x) * np.sqrt(x.size(-1))
        
        # Positional encoding
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Transformer
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Classification
        x = self.classifier(x)
        
        return x

class GenomicMLPipeline:
    """Machine learning pipeline for genomic data."""
    
    def __init__(self, model_type: str = 'cnn', device: str = 'cuda'):
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.logger = logging.getLogger(__name__)
    
    def create_model(self, vocab_size: int = 5, num_classes: int = 2, 
                    max_length: int = 1000) -> nn.Module:
        """Create model based on type."""
        if self.model_type == 'cnn':
            self.model = GenomicCNN(vocab_size=vocab_size, num_classes=num_classes, 
                                  max_length=max_length)
        elif self.model_type == 'transformer':
            self.model = GenomicTransformer(vocab_size=vocab_size, num_classes=num_classes,
                                          max_length=max_length)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.to(self.device)
        return self.model
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, lr: float = 0.001) -> List[float]:
        """Train the model."""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            accuracy = 100 * correct / total
            
            scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                self.logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, '
                               f'Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        return train_losses, val_losses
    
    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on test data."""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(target.cpu().numpy())
        
        return np.array(predictions), np.array(true_labels)
```

## 📊 Data Visualization

```python
# src/tools/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class GenomicVisualization:
    """Visualization tools for genomic data."""
    
    def __init__(self, style: str = 'seaborn'):
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_quality_metrics(self, qc_data: pd.DataFrame, output_file: str = None):
        """Plot quality control metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Per base sequence quality
        axes[0, 0].plot(qc_data['Position'], qc_data['Mean_Quality'])
        axes[0, 0].set_title('Per Base Sequence Quality')
        axes[0, 0].set_xlabel('Position in Read')
        axes[0, 0].set_ylabel('Quality Score')
        
        # Per sequence quality scores
        axes[0, 1].hist(qc_data['Quality_Score'], bins=50, alpha=0.7)
        axes[0, 1].set_title('Per Sequence Quality Scores')
        axes[0, 1].set_xlabel('Quality Score')
        axes[0, 1].set_ylabel('Count')
        
        # Sequence length distribution
        axes[1, 0].hist(qc_data['Length'], bins=50, alpha=0.7)
        axes[1, 0].set_title('Sequence Length Distribution')
        axes[1, 0].set_xlabel('Length')
        axes[1, 0].set_ylabel('Count')
        
        # GC content
        axes[1, 1].hist(qc_data['GC_Content'], bins=50, alpha=0.7)
        axes[1, 1].set_title('GC Content Distribution')
        axes[1, 1].set_xlabel('GC Content (%)')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_expression_heatmap(self, expression_data: pd.DataFrame, 
                               output_file: str = None):
        """Plot expression heatmap."""
        # Log transform and normalize
        log_data = np.log2(expression_data + 1)
        normalized_data = (log_data - log_data.mean()) / log_data.std()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(normalized_data, cmap='RdBu_r', center=0,
                   xticklabels=True, yticklabels=True)
        plt.title('Gene Expression Heatmap')
        plt.xlabel('Samples')
        plt.ylabel('Genes')
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_volcano_plot(self, de_results: pd.DataFrame, 
                         output_file: str = None):
        """Plot volcano plot for differential expression."""
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=de_results['log2FoldChange'],
            y=-np.log10(de_results['padj']),
            mode='markers',
            marker=dict(
                size=8,
                color=de_results['padj'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="P-value")
            ),
            text=de_results['gene_id'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Log2FC: %{x:.2f}<br>' +
                         '-Log10(P-value): %{y:.2f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add significance lines
        fig.add_hline(y=-np.log10(0.05), line_dash="dash", 
                     line_color="red", annotation_text="P-value = 0.05")
        fig.add_vline(x=1, line_dash="dash", line_color="blue", 
                     annotation_text="Log2FC = 1")
        fig.add_vline(x=-1, line_dash="dash", line_color="blue", 
                     annotation_text="Log2FC = -1")
        
        fig.update_layout(
            title='Volcano Plot - Differential Expression',
            xaxis_title='Log2 Fold Change',
            yaxis_title='-Log10(P-value)',
            width=800,
            height=600
        )
        
        if output_file:
            fig.write_html(output_file)
        fig.show()
    
    def plot_pca(self, expression_data: pd.DataFrame, 
                 metadata: pd.DataFrame, output_file: str = None):
        """Plot PCA of expression data."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(expression_data.T)
        
        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        # Create DataFrame
        pca_df = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'Condition': metadata['condition']
        })
        
        # Plot
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Condition',
                        title='PCA Plot - Gene Expression',
                        labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                               'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'})
        
        if output_file:
            fig.write_html(output_file)
        fig.show()
    
    def plot_variant_distribution(self, vcf_data: pd.DataFrame, 
                                 output_file: str = None):
        """Plot variant distribution."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Variant types
        variant_types = vcf_data['TYPE'].value_counts()
        axes[0, 0].pie(variant_types.values, labels=variant_types.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Variant Types')
        
        # Chromosome distribution
        chr_counts = vcf_data['CHROM'].value_counts()
        axes[0, 1].bar(range(len(chr_counts)), chr_counts.values)
        axes[0, 1].set_title('Variants per Chromosome')
        axes[0, 1].set_xlabel('Chromosome')
        axes[0, 1].set_ylabel('Number of Variants')
        axes[0, 1].set_xticks(range(len(chr_counts)))
        axes[0, 1].set_xticklabels(chr_counts.index, rotation=45)
        
        # Quality score distribution
        axes[1, 0].hist(vcf_data['QUAL'], bins=50, alpha=0.7)
        axes[1, 0].set_title('Quality Score Distribution')
        axes[1, 0].set_xlabel('Quality Score')
        axes[1, 0].set_ylabel('Count')
        
        # Allele frequency
        af_data = vcf_data['AF'].dropna()
        axes[1, 1].hist(af_data, bins=50, alpha=0.7)
        axes[1, 1].set_title('Allele Frequency Distribution')
        axes[1, 1].set_xlabel('Allele Frequency')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
```

## 📚 Learning Resources

- [Biopython Documentation](https://biopython.org/)
- [Bioconductor](https://www.bioconductor.org/)
- [GATK Best Practices](https://gatk.broadinstitute.org/hc/en-us/articles/360035894731)
- [RNA-seq Analysis](https://www.rna-seqblog.com/)

## 🔗 Upstream Source

- **Repository**: [biopython/biopython](https://github.com/biopython/biopython)
- **Bioconductor**: [bioconductor.org](https://www.bioconductor.org/)
- **GATK**: [broadinstitute/gatk](https://github.com/broadinstitute/gatk)
- **License**: BSD-3-Clause
