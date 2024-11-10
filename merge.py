import pandas as pd

# Load each dataset (assuming they are in the same folder as this script)
mutation_data = pd.read_csv('TCGA-COAD.somaticmutation_wxs.tsv', sep='\t')
expression_data = pd.read_csv('TCGA-COAD.star_tpm.tsv', sep='\t', index_col=0)
cnv_data = pd.read_csv('TCGA-COAD.gene-level_absolute.tsv', sep='\t', index_col=0)
mirna_data = pd.read_csv('TCGA-COAD.mirna.tsv', sep='\t', index_col=0)

# Set sampleID as the index for each dataset
mutation_data = mutation_data.set_index('sampleID')
expression_data = expression_data.transpose()
expression_data.index.name = 'sampleID'
cnv_data = cnv_data.transpose()
cnv_data.index.name = 'sampleID'
mirna_data = mirna_data.transpose()
mirna_data.index.name = 'sampleID'

# Debug: Print unique sample IDs in each dataset to verify overlap
print("Unique Sample IDs in Mutation Data:", mutation_data.index.unique())
print("Unique Sample IDs in Expression Data:", expression_data.index.unique())
print("Unique Sample IDs in CNV Data:", cnv_data.index.unique())
print("Unique Sample IDs in miRNA Data:", mirna_data.index.unique())

# Aggregate mutation data
mutation_summary = mutation_data.groupby('sampleID').agg({
    'gene': 'nunique',            # Count unique genes with mutations per sample
    'Amino_Acid_Change': 'nunique',  # Count unique amino acid changes per sample
    'effect': 'nunique'           # Count unique mutation effects per sample
}).rename(columns={
    'gene': 'num_genes_mutated', 
    'Amino_Acid_Change': 'num_amino_acid_changes', 
    'effect': 'num_effects'
})

# Summarize expression, miRNA, and CNV data
expression_summary = expression_data.mean(axis=1).to_frame(name='avg_expression_level')
mirna_summary = mirna_data.mean(axis=1).to_frame(name='avg_mirna_expression_level')
cnv_summary = cnv_data.mean(axis=1).to_frame(name='avg_copy_number')

# Join all datasets on sampleID
integrated_data = mutation_summary.join([
    expression_summary, 
    mirna_summary, 
    cnv_summary
], how='inner')  # Use inner join to include only samples present in all datasets

# Handle missing values
integrated_data = integrated_data.fillna(0)

# Display the integrated data
print("Integrated Dataset (Sample):")
print(integrated_data.head())

# Save the integrated dataset to a file
integrated_data.to_csv('integrated_data_for_ml.csv')
