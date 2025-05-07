#utils.py
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from numpy.linalg import eig
from numpy.linalg import norm
from scipy.optimize import minimize_scalar
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from scipy.stats import zscore
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from statsmodels.miscmodels.ordinal_model import OrderedModel
import plotly.graph_objects as go
from IPython.display import HTML
import scanpy as sc
import scipy as sp
import seaborn as sb
from scipy.sparse import csr_matrix
from scipy.io import mmread
import qnorm
import torch
import torch.nn as nn
import torch.optim as optim
import qnorm
from scipy import sparse
import numba
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
import matplotlib.gridspec as gridspec
from anndata import AnnData
from scipy.sparse import issparse
import importlib.resources
from pathlib import Path
from typing import Tuple, List
import anndata
from logging import getLogger
import time
from tqdm import tqdm  #
from itertools import product
import networkx as nx
from PyComplexHeatmap import DotClustermapPlotter,HeatmapAnnotation,anno_simple,anno_label,AnnotationBase
logger = getLogger(__name__)


def counts2FPKM(counts, genelen):
    genelen = pd.read_csv(genelen, sep=',')
    genelen['TranscriptLength'] = genelen['Transcript end (bp)'] - genelen['Transcript start (bp)']
    genelen = genelen[['Gene name', 'TranscriptLength']]
    genelen = genelen.groupby('Gene name').max()

    inter = counts.columns.intersection(genelen.index)
    if len(inter) == 0:
        raise ValueError("No overlapping genes found between counts and gene length data.")

    samplename = counts.index
    counts = counts[inter].values
    genelen = genelen.loc[inter].T.values

    totalreads = counts.sum(axis=1)
    fpkm = counts * 1e9 / (genelen * totalreads.reshape(-1, 1))
    fpkm_df = pd.DataFrame(fpkm, columns=inter, index=samplename)
    
    return fpkm_df
    
def FPKM2TPM(fpkm):
    genename = fpkm.columns
    samplename = fpkm.index
    fpkm = fpkm.values
    total = fpkm.sum(axis=1).reshape(-1, 1)
    fpkm = fpkm * 1e6 / total
    tpm = pd.DataFrame(fpkm, columns=genename, index=samplename)
    return tpm



def counts2TPM(counts, genelen):
    fpkm = counts2FPKM(counts, genelen)
    tpm = FPKM2TPM(fpkm)
    return tpm

def counts2log1tpm(adata, genelen_file=None):
    """
    Convert the count matrix of adata to TPM format (in sparse matrix mode) and perform log-normalization.

    Parameters:
    adata (AnnData): AnnData object.
    genelen_file (str): Path to the gene length file. If None, use the default file in the package.

    Returns:
    adata (AnnData): Converted AnnData object, with adata.X as a sparse matrix.
    """
    if genelen_file is None:
        with importlib.resources.path("Chunk.data", "GeneLength.txt") as default_path:
            genelen_file = default_path

    if isinstance(genelen_file, Path):
        genelen_file = str(genelen_file)

    adata = adata[:, ~adata.var_names.duplicated()].copy()

    counts = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)

    tpm = counts2TPM(counts, genelen_file)

    genes_in_tpm = tpm.columns

    adata = adata[:, adata.var_names.isin(genes_in_tpm)].copy()

    tpm = tpm.loc[:, adata.var_names]

    tpm_sparse = csr_matrix(tpm.values)

    adata.X = tpm_sparse
    adata.var_names = tpm.columns
    adata.obs_names = tpm.index

    sc.pp.log1p(adata)

    return adata


def select_z_score_lr(H_array,comm_matrix,threshold=2):
    df = pd.DataFrame(H_array,index = comm_matrix.index,columns=['loading'])
    df_filtered = df[df['loading'] != 0].copy()
    df_filtered['z_score'] = zscore(df_filtered['loading'])
    return df_filtered[df_filtered['z_score'] > threshold]

@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1,t2):
	sum=0
	for i in range(t1.shape[0]):
		sum+=(t1[i]-t2[i])**2
	return np.sqrt(sum)

@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
	n=X.shape[0]
	adj=np.empty((n, n), dtype=np.float32)
	for i in numba.prange(n):
		for j in numba.prange(n):
			adj[i][j]=euclid_dist(X[i], X[j])
	return adj

def calculate_adj_matrix(x, y, x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, histology=True):
	#x,y,x_pixel, y_pixel are lists
	if histology:
		assert (x_pixel is not None) & (x_pixel is not None) & (image is not None)
		assert (len(x)==len(x_pixel)) & (len(y)==len(y_pixel))
		print("Calculateing adj matrix using histology image...")
		#beta to control the range of neighbourhood when calculate grey vale for one spot
		#alpha to control the color scale
		beta_half=round(beta/2)
		g=[]
		for i in range(len(x_pixel)):
			max_x=image.shape[0]
			max_y=image.shape[1]
			nbs=image[max(0,x_pixel[i]-beta_half):min(max_x,x_pixel[i]+beta_half+1),max(0,y_pixel[i]-beta_half):min(max_y,y_pixel[i]+beta_half+1)]
			g.append(np.mean(np.mean(nbs,axis=0),axis=0))
		c0, c1, c2=[], [], []
		for i in g:
			c0.append(i[0])
			c1.append(i[1])
			c2.append(i[2])
		c0=np.array(c0)
		c1=np.array(c1)
		c2=np.array(c2)
		c3=(c0*np.var(c0)+c1*np.var(c1)+c2*np.var(c2))/(np.var(c0)+np.var(c1)+np.var(c2))
		c4=(c3-np.mean(c3))/np.std(c3)
		z_scale=np.max([np.std(x), np.std(y)])*alpha
		z=c4*z_scale
		z=z.tolist()
		X=np.array([x, y, z]).T.astype(np.float32)
	else:
		print("Calculateing adj matrix using xy only...")
		X=np.array([x, y]).T.astype(np.float32)
	return pairwise_distance(X)

def create_knn_adj(adj, k=20):
    """
    Construct an adjacency graph using the k-nearest neighbors algorithm and return a sparse matrix.

    Parameters:
    adj (np.ndarray or scipy.sparse matrix): Input adjacency matrix.
    k (int): Number of nearest neighbors, default is 20.

    Returns:
    adj_sparse (scipy.sparse.csr_matrix): Processed sparse adjacency matrix.
    """
    adj_sparse = kneighbors_graph(adj, k, mode='connectivity')

    return adj_sparse

def getLRcomm(W, H,comm_matrix, patterns,lr_expr_pairs,position = None,zscore_threshold = 2):
    phe_factor_sampleW = {}
    phe_factor_cci = {}
    phe_factor_cciW = {}
    for i in patterns:
        if position is not None:
            phe_factor_sampleW[i] = np.mean(W[:, i-1][position[0]:position[1]+1])
        else:
            phe_factor_sampleW[i] = np.mean(W[:, i-1])
        phe_factor_cci[i] = select_z_score_lr(H[i-1],comm_matrix,threshold=zscore_threshold)
        phe_factor_cciW[i] = phe_factor_sampleW[i] * phe_factor_cci[i]
    final_df = merge_loading_dfs(phe_factor_cciW)
    final_df = final_df.reset_index().rename(columns={'index': 'l-r'}) 
    
    result = pd.merge(final_df, lr_expr_pairs[lr_expr_pairs['l-r'].isin(final_df['l-r'])], left_on='l-r', right_on='l-r', how='inner')
    result = result[['l-r','ligand','receptor','loading']]
    
    return result

def merge_loading_dfs(df_dict):
    all_indexes = pd.Index([])  
    for df in df_dict.values():
        all_indexes = all_indexes.union(df.index)  
    final_df = pd.DataFrame(index=all_indexes)

    for df in df_dict.values():
        final_df['loading'] = final_df.get('loading', 0)  
        final_df['loading'] += df['loading'].reindex(final_df.index, fill_value=0)

    return final_df.sort_values(by='loading', ascending=False)


def Singleassociationanalysis(W, clin_df, phenotype_types, covariates=None, alpha=0.05):
    """
    Perform regression analysis between the NMF pattern weight matrix W and multiple phenotypes (For single element).

    Parameters:
    - W: np.ndarray or pd.DataFrame, (samples, patterns) NMF pattern weight matrix
    - clin_df: pd.DataFrame, (samples, phenotypes) DataFrame containing patient phenotype data
    - phenotype_types: dict, specifying the variable type for each phenotype {"phenotype_name": "binary" or "continuous" or "ordinal"}
    - covariates: pd.DataFrame or None, (samples, covariate_num) Optional covariates (e.g., age, gender)
    - alpha: float, default 0.05, significance threshold after multiple correction

    Returns:
    - results_df: pd.DataFrame, regression analysis results, including regression coefficients, P-values, and FDR-adjusted P-values
    """
    if isinstance(W, np.ndarray):
        W = pd.DataFrame(W, columns=[f"Pattern_{i+1}" for i in range(W.shape[1])], index=clin_df.index)

    results = []

    # Iterate over each phenotype
    for phe in list(phenotype_types.keys()):
        if phe not in clin_df.columns:
            print(f"Warning: {phe} not in clin_df 中, skipping this variable.")
            continue

        # dropna
        valid_idx = clin_df[phe].dropna().index
        Y = clin_df.loc[valid_idx, phe]
        W_valid = W.loc[valid_idx, :]
        cov_valid = covariates.loc[valid_idx, :] if covariates is not None else None

        # Iterate each patterns
        for mode in W_valid.columns:
            X = W_valid[[mode]]  
            if cov_valid is not None:
                X = pd.concat([X, cov_valid], axis=1)

            if phenotype_types[phe] == "continuous":
                X = sm.add_constant(X)  
                model = sm.OLS(Y, X).fit()
                coef = model.params.iloc[1]
                p_value = model.pvalues.iloc[1]
                r_squared = model.rsquared
            elif phenotype_types[phe] == "binary":
                X = sm.add_constant(X)  
                model = sm.Logit(Y, X).fit(disp=0)
                coef = model.params.iloc[1]
                p_value = model.pvalues.iloc[1]
                r_squared = model.prsquared
            elif phenotype_types[phe] == "ordinal":
                Y = Y.astype("category") 
                Y = Y.cat.set_categories(sorted(Y.unique()), ordered=True)
                model = OrderedModel(Y, X, distr="logit").fit(method="bfgs", disp=False)
                coef = model.params.iloc[0]
                p_value = model.pvalues.iloc[0]
                r_squared = None

            else:
                print(f"Error: Unsupported phenotype type {phenotype_types[phe]}，Skipping {phe}.")
                continue

            results.append({"Phenotype": phe, "Mode": mode, "Coefficient": coef, "P_value": p_value, "R_squared": r_squared})

    results_df = pd.DataFrame(results)

    # **FDR **
    if not results_df.empty:
        results_df["Adjusted_P_value"] = np.nan  

        for phe in results_df["Phenotype"].unique():
            phe_mask = results_df["Phenotype"] == phe
            raw_p_values = results_df.loc[phe_mask, "P_value"].values
            _, corrected_p_values, _, _ = multipletests(raw_p_values, method="fdr_bh")
            results_df.loc[phe_mask, "Adjusted_P_value"] = corrected_p_values

    else:
        print("Not enough data for regression analysis.")
        return None
    return results_df

def Covariateassociationanalysis(W, clin_df, phenotype_types, covariates, alpha=0.05):
    """
    Perform multivariate regression analysis between the NMF pattern weight matrix W and clinical phenotypes (with covariates).

    Parameters:
    - W: np.ndarray or pd.DataFrame, (samples, patterns) NMF pattern weight matrix
    - clin_df: pd.DataFrame, (samples, features) Clinical data containing phenotypes and covariates
    - phenotype_types: dict, specifying the types of phenotypes of interest (continuous, binary, ordinal)
    - covariates: dict, specifying the covariates and their types (continuous, binary, ordinal)
    - alpha: float, default 0.05, significance threshold after multiple correction

    Returns:
    - results_df: pd.DataFrame, multivariate regression results, including regression coefficients, P-values, and FDR-adjusted P-values
    """
    if isinstance(W, np.ndarray):
        W = pd.DataFrame(W, columns=[f"Pattern_{i+1}" for i in range(W.shape[1])],index=clin_df.index)

    results = []

    for phenotype, p_type in phenotype_types.items():
        if phenotype not in clin_df.columns:
            continue  
        

        relevant_cols = [phenotype] + covariates
        df = clin_df.join(W, how="inner")  
        df = df[relevant_cols + list(W.columns)].dropna().copy()  

        if df.empty:
            continue

        for mode in W.columns:
            X = df[[mode] + covariates]  
            Y = df[phenotype]  
            X = sm.add_constant(X)  

            try:
                if p_type == "continuous":
                    model = sm.OLS(Y, X).fit()
                elif p_type == "binary":
                    model = Logit(Y, X).fit(disp=0)
                elif p_type == "ordinal":
                    model = OrderedModel(Y, X, distr="logit").fit(method="bfgs", disp=0)
                else:
                    raise ValueError(f"Unsupported phenotype type: {p_type}")

                coef = model.params.iloc[1]  
                p_value = model.pvalues.iloc[1]  

                results.append({"Phenotype": phenotype, "Mode": mode, "Coefficient": coef, "P_value": p_value})

            except Exception as e:
                print(f"Error in model fitting for {phenotype} - {mode}: {e}")
                continue


    results_df = pd.DataFrame(results)

    # FDR
    if not results_df.empty:
        for phenotype in phenotype_types.keys():
            mask = results_df["Phenotype"] == phenotype
            if mask.sum() > 0:
                _, adj_p_values, _, _ = multipletests(results_df.loc[mask, "P_value"], method="fdr_bh")
                results_df.loc[mask, "Adjusted_P_value"] = adj_p_values
    else:
        print("Not enough data for regression analysis.")
        return None

    return results_df

"""
def plot_significance_heatmap(results_df, alpha, savefig=None):

    Plot a significance heatmap using -log10(Adjusted P value) to display significance.
 

    results_df["log10_p_adjust"] = -np.log10(results_df["Adjusted_P_value"])

    heatmap_data = results_df.pivot(index="Phenotype", columns="Mode", values="log10_p_adjust")
    significance = results_df.pivot(index="Phenotype", columns="Mode", values="Adjusted_P_value") < alpha

    significance_labels = significance.replace({True: '*', False: ''})
    
    plt.figure(figsize=(10, 1.5))
    ax = sns.heatmap(heatmap_data, annot=significance_labels, fmt="",cmap="coolwarm", center=0, linewidths=0.5,
        annot_kws={"fontsize": 12, "color": "black"},  
        cbar_kws={"shrink": 0.8}  )
    
    plt.title("Significance Heatmap of NMF Patterns & Phenotypes (-log10(Adjusted P))")
    plt.xlabel("NMF Pattern")
    plt.ylabel("Phenotype")
    
    cbar = ax.collections[0].colorbar
    cbar.set_label("-log10(Adjusted P value)")
    if savefig is not None:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
    plt.show()
"""

def plot_significance_heatmap(results_df, alpha=0.05, savefig=None):
    """
    Plot a significance heatmap using -log10(Adjusted P value) to display significance.
    
    Parameters:
    - results_df: DataFrame containing 'Phenotype', 'Mode', 'Adjusted_P_value', and 'Coefficient'.
    - alpha: Significance threshold for adjusted P-values.
    - savefig: If provided, saves the figure to the specified path.
    """
    results_df["log10_p_adjust"] = -np.log10(results_df["Adjusted_P_value"])
    
    results_df["Color"] = np.where(results_df["Coefficient"] > 0, results_df["log10_p_adjust"], -results_df["log10_p_adjust"])

    heatmap_data = results_df.pivot(index="Phenotype", columns="Mode", values="Color")
    significance = results_df["Adjusted_P_value"] < alpha

    significance_labels = results_df.pivot(index="Phenotype", columns="Mode", values="Adjusted_P_value")
    significance_labels = significance_labels < alpha
    significance_labels = significance_labels.replace({True: '*', False: ''})

    num_phenotypes = len(heatmap_data)
    figsize_height = 1.5 * num_phenotypes
    
    plt.figure(figsize=(10, figsize_height))
    ax = sns.heatmap(heatmap_data, annot=significance_labels, fmt="", cmap="bwr", center=0, linewidths=0.5,
                     annot_kws={"fontsize": 12, "color": "black"}, cbar_kws={"shrink": 0.8})
    
    plt.title("Significance Heatmap of Patterns & Phenotypes, Red: Positive, Blue: Negative)")
    plt.xlabel("Pattern")
    plt.ylabel("Phenotype")
    
    cbar = ax.collections[0].colorbar
    cbar.set_label("-log10(Adjusted P value)")
    
    if savefig is not None:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sankey(
    res, 
    title="Ligand-Receptor", 
    width=1000, 
    height=800, 
    pad=100, 
    thickness=40, 
    font_size=15, 
    node_colors=None, 
    link_colors=None, 
    line_color="black", 
    line_width=0.5
):
    """
    Generate an interactive Sankey diagram for Ligand-Receptor interactions.

    Parameters:
    res (pd.DataFrame): DataFrame containing columns 'ligand', 'receptor', and 'loading'.
    title (str): Title of the diagram, default is "Ligand-Receptor".
    width (int): Width of the diagram, default is 1000.
    height (int): Height of the diagram, default is 800.
    pad (int): Spacing between nodes, default is 100.
    thickness (int): Thickness of the nodes, default is 40.
    font_size (int): Font size, default is 15.
    node_colors (list): List of colors for nodes, default is None (blue for ligands, green for receptors).
    link_colors (list): List of colors for links, default is None (semi-transparent green).
    line_color (str): Border color of nodes, default is "black".
    line_width (float): Border width of nodes, default is 0.5.

    Returns:
    HTML: HTML object of the interactive Sankey diagram.
    """
    ligands = res['ligand'].unique()
    receptors = res['receptor'].unique()

    nodes = list(ligands) + list(receptors)

    node_indices = {node: i for i, node in enumerate(nodes)}

    source = []
    target = []
    value = []

    for _, row in res.iterrows():
        source.append(node_indices[row['ligand']])  # ligand indexs
        target.append(node_indices[row['receptor']])  # receptor indexs
        value.append(row['loading'])  # 权重

    if node_colors is None:
        node_colors = ["#7495D3"] * len(ligands) + ["#C798EE"] * len(receptors)

    if link_colors is None:
        link_colors = ["#D1D1D1"] * len(source) 


    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=pad,  
            thickness=thickness,  
            line=dict(color=line_color, width=line_width), 
            label=nodes,
            color=node_colors  
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors  
        )
    )])


    fig.update_layout(
        title_text=title,
        font_size=font_size,  
        width=width,  
        height=height  
    )

    return HTML(fig.to_html())

def calculate_correlation_matrix(bulk_lr_expr_df, sc_lr_expr_df):
    """
    Calculate the correlation matrix between bulk data and single-cell data.

    Parameters:
    bulk_lr_expr_df (DataFrame): Expression matrix of bulk data with ligand-receptor gene.
    sc_lr_expr_df (DataFrame): Expression matrix of single-cell data.

    Returns:
    X (np.ndarray): Correlation matrix between bulk data and single-cell data.
    """
    bulk_lr_expr_df = bulk_lr_expr_df.loc[sc_lr_expr_df.index]

    dataset0 = pd.concat([bulk_lr_expr_df, sc_lr_expr_df], axis=1).astype(np.float64)  

    dataset1 = qnorm.quantile_normalize(dataset0, axis=1)
    dataset1 = pd.DataFrame(dataset1, index=dataset0.index, columns=dataset0.columns)

    n_bulk_samples = bulk_lr_expr_df.shape[1]
    Expression_bulk = dataset1.iloc[:, :n_bulk_samples]  
    Expression_cell = dataset1.iloc[:, n_bulk_samples:]  

    bulk_mean = Expression_bulk.mean(axis=0)
    bulk_std = Expression_bulk.std(axis=0)
    bulk_std[bulk_std == 0] = 1e-10
    
    Expression_bulk_z = (Expression_bulk - bulk_mean) / bulk_std
    
    cell_mean = Expression_cell.mean(axis=0)
    cell_std = Expression_cell.std(axis=0)

    cell_std[cell_std == 0] = 1e-10

    Expression_cell_z = (Expression_cell - cell_mean) / cell_std
    X = np.dot(Expression_bulk_z.T, Expression_cell_z) / (Expression_bulk.shape[0] - 1)

    quality_check = np.percentile(X, [0, 25, 50, 75, 100])

    print("|**************************************************|")
    print("Performing quality-check for the correlations")
    print("The five-number summary of correlations:")
    print(f"Min: {quality_check[0]}")
    print(f"25th Percentile: {quality_check[1]}")
    print(f"Median: {quality_check[2]}")
    print(f"75th Percentile: {quality_check[3]}")
    print(f"Max: {quality_check[4]}")
    print("|**************************************************|")

    if quality_check[2] < 0.1:
        print("Warning: The median correlation between the single-cell and bulk samples is relatively low.")

    return X

def similarity2adjacent(adata, key=None):
    """
    Calculate the cell adjacency matrix based on the cell similarity sparse matrix in adata.obsp.

    Parameters:
    adata (AnnData): AnnData object.
    key (str): Key in adata.obsp.

    Returns:
    connectivities (scipy.sparse.csr_matrix): Cell adjacency matrix derived from the cell similarity sparse matrix.
    """
    connectivities = adata.obsp[key]

    connectivities.setdiag(0)

    connectivities.data = np.ones_like(connectivities.data)

    return connectivities


def center_features(X):
    feature_means = X.mean(dim=0, keepdim=True)
    X_centered = X - feature_means
    return X_centered, feature_means


def getPosipotentialCCI(adata, model, threshold_percent=60, savefig=None, embedding='umap'):
    """
    Generate a UMAP, t-SNE, or other embedding plot based on the threshold percentile and mark the top cells.

    Parameters:
    - adata (AnnData): AnnData object.
    - model: Model object containing linear weights (beta).
    - threshold_percent (float): Threshold percentile, default is 60 (i.e., top 40%).
    - savefig (str): Path to save the figure, default is None (do not save).
    - embedding (str): Embedding to use for plotting (e.g., 'umap', 'tsne'). Default is 'umap'.

    Returns:
    No return value, directly display or save the plot.
    """
    # Extract linear weights (beta) from the model
    beta = model.linear.weight.detach().cpu().squeeze().numpy()
    
    # Calculate the threshold based on the given percentile
    threshold = np.percentile(beta, threshold_percent)
    
    # Mark cells above the threshold
    marked = np.where((beta >= threshold) & (beta > 0), 1, 0)
    
    # Add the marked cells to adata.obs
    adata.obs['phe_cell'] = marked.astype(str)
    
    # Define the color palette for highlighting
    highlight_palette = {'0': 'lightgrey', '1': 'red'}
    
    # Check if the specified embedding exists in adata.obsm
    if f'X_{embedding}' not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm. Please run the corresponding dimensionality reduction first.")
    
    # Plot the embedding with highlighted cells
    sc.pl.embedding(
        adata,
        basis=embedding,
        color='phe_cell',
        palette=highlight_palette,
        show=False,
        return_fig=True,
        title='Phenotype associated CCI active cells'
    )
    
    # Add a title to the plot
    #plt.title(f"{embedding.upper()} Plot with Top {100 - threshold_percent}% Cells Highlighted")
    
    # Save the figure if savefig is provided
    if savefig is not None:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {savefig}")
    
    # Show the plot
    plt.show()

def getNegapotentialCCI(adata, model, threshold_percent=40, savefig=None, embedding='umap'):
    """
    Generate a UMAP, t-SNE, or other embedding plot based on the threshold percentile and mark the most negatively correlated cells.

    Parameters:
    - adata (AnnData): AnnData object.
    - model: Model object containing linear weights (beta).
    - threshold_percent (float): Threshold percentile, default is 40 (i.e., most negatively correlated 40%).
    - savefig (str): Path to save the figure, default is None (do not save).
    - embedding (str): Embedding to use for plotting (e.g., 'umap', 'tsne'). Default is 'umap'.

    Returns:
    No return value, directly display or save the plot.
    """
    # Extract linear weights (beta) from the model
    beta = model.linear.weight.detach().cpu().squeeze().numpy()
    
    # Filter negative beta values
    negative_beta = beta[beta < 0]
    
    # Calculate the threshold based on the given percentile
    threshold = np.percentile(negative_beta, threshold_percent)
    
    # Mark cells below the threshold (most negatively correlated)
    marked = np.where((beta < 0) & (beta <= threshold), 1, 0)
    
    # Add the marked cells to adata.obs
    adata.obs['negative_phe_cell'] = marked.astype(str)
    
    # Define the color palette for highlighting
    highlight_palette = {'0': 'lightgrey', '1': 'blue'}
    
    # Check if the specified embedding exists in adata.obsm
    if f'X_{embedding}' not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm. Please run the corresponding dimensionality reduction first.")
    
    # Plot the embedding with highlighted cells
    sc.pl.embedding(
        adata,
        basis=embedding,
        color='negative_phe_cell',
        palette=highlight_palette,
        show=False,
        return_fig=True,
        title='Phenotype associated CCI active cells'
    )
    
    # Add a title to the plot
    #plt.title(f"{embedding.upper()} Plot with Most Negatively Correlated {threshold_percent}% Cells Highlighted")
    
    # Save the figure if savefig is provided
    if savefig is not None:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {savefig}")
    
    # Show the plot
    plt.show()

def getCCcomm(adata, gene1, gene2, L_threshold=20, R_threshold=20, marked_col='phe_cell', 
             savefig=None, saveflag=True, size_factor=None, embedding='umap'):
    """
    Mark cells with high expression of two genes on a specified embedding plot, limited to cells where marked is 1.

    Parameters:
    - adata: AnnData object.
    - gene1: Name of the first gene (e.g., Ligand).
    - gene2: Name of the second gene (e.g., Receptor).
    - L_threshold: High expression percentile threshold for the first gene (default is top 20%).
    - R_threshold: High expression percentile threshold for the second gene (default is top 20%).
    - marked_col: Column name for the marked column.
    - savefig: Filename to save the figure.
    - saveflag: Whether to delete the temporary annotation after plotting.
    - size_factor: Scaling factor for point sizes. If None, default sizes are used.
    - embedding: Embedding to use for plotting (e.g., 'umap', 'tsne', 'pca'). Default is 'umap'.
    """

    if f'X_{embedding}' not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm. Please run {embedding.upper()} first.")
    
    marked_cells = adata.obs[marked_col] == '1' if marked_col else np.ones(adata.n_obs, dtype=bool)
    
    gene1_expression = adata[:, gene1].X.toarray().flatten()
    gene2_expression = adata[:, gene2].X.toarray().flatten()
    
    gene1_threshold = np.percentile(gene1_expression, 100 - L_threshold)
    gene2_threshold = np.percentile(gene2_expression, 100 - R_threshold)
    
    gene1_high = (gene1_expression >= gene1_threshold) & (gene1_expression > 0)
    gene2_high = (gene2_expression >= gene2_threshold) & (gene2_expression > 0)
    
    adata.obs[f'{gene1}_{gene2}'] = 'Background'
    combined_markers = np.select(
        [gene1_high & gene2_high, gene1_high, gene2_high],
        ['Autocrine', gene1, gene2],
        default='Background'
    )
    adata.obs.loc[marked_cells, f'{gene1}_{gene2}'] = combined_markers[marked_cells]
    
    highlight_palette = {
        gene1: 'blue', 
        gene2: 'red',  
        'Autocrine': 'purple',  
        'Background': 'lightgrey'  
    }
    
    size_params = {}
    if size_factor is not None:
        fig = sc.pl.embedding(adata, basis=embedding, return_fig=True)  
        default_sizes = fig.axes[0].collections[0].get_sizes()
        default_size = np.mean(default_sizes) if len(default_sizes) > 0 else 20
        plt.close(fig)
        
        adata.obs['point_size'] = default_size
        for category in [gene1, gene2, 'Autocrine']:
            adata.obs.loc[adata.obs[f'{gene1}_{gene2}'] == category, 'point_size'] *= size_factor
        size_params["size"] = adata.obs['point_size']
    
    sc.pl.embedding(
        adata, 
        basis=embedding, 
        color=[f'{gene1}_{gene2}'], 
        palette=highlight_palette, 
        title=f'{gene1}-{gene2}',
        return_fig=True,
        **size_params
    )
    
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {savefig}")
    
    if not saveflag:
        del adata.obs[f'{gene1}_{gene2}']
        if 'point_size' in adata.obs: 
            del adata.obs['point_size']
    
    plt.show()

def getPatternDistribution(W, labels, phenotype_interval, savefig=None):
    """
    Plot violin plots to show the distribution of each feature in the W matrix across binary phenotype samples.

    Parameters:
    W (np.ndarray): Input matrix with shape (n_samples, n_features).
    labels (list): Category labels, default is ['Normal', 'Tumor'].
    phenotype_interval (dict): Boundary points for different types of samples.
    savefig (str): Path to save the generated image, default is None (do not save).
    """
    num_features = W.shape[1]  
    num_subplots = int(np.ceil(np.sqrt(num_features)))  
    num_rows = num_subplots
    num_cols = num_subplots if num_features % num_subplots == 0 else num_subplots + 1  

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))  
    axes = axes.flatten() 
    plot_color=["#F56867","#FEB915","#C798EE","#7495D3","#3A84E6","#DB4C6C","#F9BD3F","#DAB370","#877F6C","#268785"]
    for i in range(num_features):
        ax = axes[i]
        p = []
        for j in range(len(phenotype_interval)):
            p.append(W[:, i][phenotype_interval[j+1][0]:phenotype_interval[j+1][1]+1])
        #sns.violinplot(ax=ax, data=p, palette=plot_color[:len(phenotype_interval)])
        sns.violinplot(ax=ax, data=p, palette='Set2')
        ax.set_title(f"Pattern {i + 1}")
        ax.set_xticks(list(range(len(phenotype_interval))))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Values")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout() 

    if savefig is not None:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {savefig}")

    plt.show()

def refine_beta(sample_id, pred, dis, shape="hexagon", method="mean"):
    """
    Refine the prediction results of continuous variables by considering the neighborhood information.

    Parameters:
    - sample_id: Unique identifier for each spot (e.g., barcode).
    - pred: Initial prediction results for each spot (continuous variable).
    - dis: Distance matrix between spots.
    - shape: Spatial arrangement of spots, can be "hexagon" (hexagonal) or "square" (square).
    - method: Neighborhood adjustment method, supports "mean" (mean) or "median" (median).

    Returns:
    - refined_pred: Refined prediction results.
    """
    refined_pred = []
    pred = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
    
    # Determine the number of neighbors
    if shape == "hexagon":
        num_nbs = 6  # Hexagonal arrangement, 6 neighbors
    elif shape == "square":
        num_nbs = 4  # Square arrangement, 4 neighbors
    else:
        print("Shape not recognized, shape='hexagon' for Visium data, 'square' for ST data.")
        return
    
    for i in range(len(sample_id)):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :].sort_values()  
        nbs = dis_tmp[0:num_nbs + 1]  
        nbs_pred = pred.loc[nbs.index, "pred"]  
        
        if method == "mean":
            refined_pred.append(nbs_pred.mean())  
        elif method == "median":
            refined_pred.append(nbs_pred.median())  
        else:
            raise ValueError("Method not recognized. Use 'mean' or 'median'.")
    
    return refined_pred

def plot_Spatiallr(
    adata: AnnData,
    lr_pair: tuple[str, str],
    layer: str = None,
    topn_frac: float = 0.2,
    knn: int = 8,
    pt_size: float = 2.0,
    alpha_min: float = 0.1,
    max_cut: float = 0.95,
    figsize: tuple = (12, 6),
    dual_plot: bool = True
) -> plt.Figure:
    """
    Visualize the spatial co-localization of a ligand-receptor pair.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing spatial transcriptomics data.
    lr_pair : tuple[str, str]
        Ligand-receptor pair names, e.g., ('Ptn', 'Ptprz1').
    layer : str, optional
        Use a specific matrix from adata.layers; default is adata.X.
    topn_frac : float, optional
        Proportion of cells considered as high expression (default: 0.2).
    knn : int, optional
        Number of nearest neighbors (default: 8).
    pt_size : float, optional
        Size of scatter points (default: 2).
    alpha_min : float, optional
        Minimum transparency (default: 0.1).
    max_cut : float, optional
        Maximum cutoff for LR activity (default: 0.95).
    figsize : tuple, optional
        Size of the figure (default: (12, 6)).
    dual_plot : bool, optional
        Whether to show both expression and activity plots (default: True).

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the visualization results.
    """
    if lr_pair[0] not in adata.var_names or lr_pair[1] not in adata.var_names:
        raise ValueError("Ligand or receptor not found in gene names")
    
    expr = adata.layers[layer] if layer else adata.X
    if issparse(expr):
        expr = expr.toarray()  
    
    if 'spatial' not in adata.obsm:
        raise KeyError("Spatial coordinates not found in adata.obsm['spatial']")
    location = adata.obsm['spatial']
    
    adata2 = adata[adata.obs['refined_beta'] > 0] 
    expr2 = expr[adata.obs['refined_beta'] > 0]
    location2 = location[adata.obs['refined_beta'] > 0]
    
    nn_model = NearestNeighbors(n_neighbors=knn+1).fit(location2)
    _, nn_indices = nn_model.kneighbors(location2)
    
    lig_idx = adata.var_names.get_loc(lr_pair[0])
    rec_idx = adata.var_names.get_loc(lr_pair[1])
    ligand = expr2[:, lig_idx]
    receptor = expr2[:, rec_idx]
    
    neighbor_expr = np.zeros((2, expr2.shape[0]))
    for i in range(expr2.shape[0]):
        neighbors = nn_indices[i, 1:]
        neighbor_expr[0, i] = np.max(ligand[neighbors])
        neighbor_expr[1, i] = np.max(receptor[neighbors])
    
    lr_activity = np.maximum(ligand * neighbor_expr[1], receptor * neighbor_expr[0])
    lr_cut = np.quantile(lr_activity, max_cut)
    lr_activity = np.clip(lr_activity, None, lr_cut)
    
    n_cells = expr2.shape[0]
    topn = int(topn_frac * n_cells)
    
    lig_order = np.argsort(-ligand + np.random.randn(n_cells) * 1e-6)
    lig_high = lig_order[:topn] if np.sum(ligand > 0) >= topn else np.where(ligand > 0)[0]
    rec_order = np.argsort(-receptor + np.random.randn(n_cells) * 1e-6)
    rec_high = rec_order[:topn] if np.sum(receptor > 0) >= topn else np.where(receptor > 0)[0]
    
    exp_type = np.zeros(n_cells, dtype=int)
    exp_type[lig_high] = 1
    exp_type[rec_high] = 2
    exp_type[np.intersect1d(lig_high, rec_high)] = 3
    
    plot_df = pd.DataFrame({
        'x': location2[:, 0],
        'y': location2[:, 1],
        'type': exp_type,
        'activity': lr_activity
    })
    
    single_width = figsize[0] / 2
    if dual_plot:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2]) 
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
    else:
        fig, ax2 = plt.subplots(1, 1, figsize=(single_width*1.2, figsize[1]))
    
    if dual_plot:
        ax1.scatter(location[:, 0], location[:, 1], color='gray', alpha=0.2, s=pt_size)
        colors = ['gray', 'red', 'green', 'blue']
        labels = ['Both low', 'Ligand high', 'Receptor high', 'Both high']
        for i in range(4):
            mask = plot_df['type'] == i
            ax1.scatter(
                plot_df.loc[mask, 'x'], 
                plot_df.loc[mask, 'y'], 
                c=colors[i], 
                label=labels[i],
                s=pt_size
            )
        ax1.legend(loc='best')
        ax1.set_title(f"{lr_pair[0]}-{lr_pair[1]} Expression")
        ax1.invert_yaxis()
    
    alpha = (lr_activity - lr_activity.min()) / (lr_activity.max() - lr_activity.min()) * (1 - alpha_min) + alpha_min
    ax2.scatter(location[:, 0], location[:, 1], color='gray', alpha=0.1, s=pt_size)
    scatter = ax2.scatter(
        plot_df['x'], 
        plot_df['y'], 
        c=plot_df['activity'], 
        cmap='RdGy_r',
        s=pt_size,
        alpha=alpha
    )
    plt.colorbar(scatter, ax=ax2, label='LR Activity', shrink=0.8) 
    ax2.set_title(f"{lr_pair[0]}-{lr_pair[1]} Activity")
    ax2.invert_yaxis()
    
    plt.tight_layout()
    return fig

def CCIdetect(
    adata: anndata.AnnData,          
    celltype_key: str,               
    interactions: pd.DataFrame,      
    senders: List[str] = None,       
    receivers: List[str] = None,     
    iterations: int = 1000,          
    threshold: float = 0.1,          
    pvalue_threshold: float = 0.05,  
    subsampling_fraction: float = 1.0  
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
   
    Parameters:
        adata: AnnData object containing count data and metadata
        celltype_key: Column name in adata.obs representing cell types
        interactions: DataFrame with columns 'ligand' and 'receptor'
        senders: List of sender cell types (subset), default None for all types
        receivers: List of receiver cell types (subset), default None for all types
        iterations: Number of randomizations
        threshold: Expression percentage threshold
        pvalue_threshold: p-value significance threshold
        subsampling_fraction: Downsampling ratio (0 < fraction ≤ 1). The default value 1.0 indicates that no sample is collected
    
    Returns:
        pvalues: p-value table
        means: geometric mean table
        significant_means: significant geometric mean table
    """
    
    logger.info(f"Running CCIdetect: iterations={iterations}, threshold={threshold}, "
                f"pvalue_threshold={pvalue_threshold}, senders={senders}, receivers={receivers}, "
                f"subsampling_fraction={subsampling_fraction}")
    
    if not 0 < subsampling_fraction <= 1:
        raise ValueError("subsampling_fraction must be between 0 and 1")
    
    start_time = time.time()
    meta = pd.DataFrame({
        'cell': adata.obs.index,
        'cell_type': adata.obs[celltype_key]
    })
    counts = pd.DataFrame(
        adata.X.T.toarray() if hasattr(adata.X.T, 'toarray') else adata.X,  
        index=adata.var.index,
        columns=adata.obs.index
    ).astype('float32')
    logger.info(f"Data extraction time: {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    common_cells = sorted(set(meta['cell']).intersection(counts.columns))
    meta = meta[meta['cell'].isin(common_cells)].set_index('cell')
    
    all_cluster_names = sorted(meta['cell_type'].unique())
    senders = senders if senders is not None else all_cluster_names
    receivers = receivers if receivers is not None else all_cluster_names
    
    invalid_senders = set(senders) - set(all_cluster_names)
    invalid_receivers = set(receivers) - set(all_cluster_names)
    if invalid_senders:
        raise ValueError(f"Invalid sender cell types: {invalid_senders}")
    if invalid_receivers:
        raise ValueError(f"Invalid receiver cell types: {invalid_receivers}")
    
    meta = meta[meta['cell_type'].isin(senders + receivers)]
    
    if subsampling_fraction < 1.0:
        sampled_cells = meta.sample(frac=subsampling_fraction, random_state=42).index
        meta = meta.loc[sampled_cells]
        counts = counts[sampled_cells]
        logger.info(f"Subsampled to {len(sampled_cells)} cells (fraction={subsampling_fraction})")
    else:
        counts = counts[meta.index]
    
    cluster_names = sorted(set(senders).union(receivers))
    
    valid_genes = set(counts.index)
    interactions = interactions[
        interactions['ligand'].isin(valid_genes) & 
        interactions['receptor'].isin(valid_genes)
    ].copy()
    if interactions.empty:
        raise ValueError("No valid ligand-receptor pairs found in counts data.")
    
    interactions['interaction'] = interactions['ligand'] + '_' + interactions['receptor']
    interactions.set_index('interaction', inplace=True)
    logger.info(f"Preprocessing time: {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    clusters_means = np.zeros((len(counts.index), len(cluster_names)), dtype='float32')
    for i, cluster in enumerate(cluster_names):
        cells = meta[meta['cell_type'] == cluster].index
        clusters_means[:, i] = counts[cells].mean(axis=1).values
    clusters_means = pd.DataFrame(clusters_means, index=counts.index, columns=cluster_names)
    logger.info(f"Cluster means time: {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    cluster_pairs = [(s, r) for s in senders for r in receivers]
    pair_columns = [f"{s}_{r}" for s, r in cluster_pairs]
    
    ligand_means = clusters_means.loc[interactions['ligand']].values
    receptor_means = clusters_means.loc[interactions['receptor']].values
    means = np.zeros((len(interactions), len(cluster_pairs)), dtype='float32')
    for j, (s, r) in enumerate(cluster_pairs):
        s_idx, r_idx = cluster_names.index(s), cluster_names.index(r)
        means[:, j] = np.where(
            (ligand_means[:, s_idx] == 0) | (receptor_means[:, r_idx] == 0),
            0,
            np.sqrt(ligand_means[:, s_idx] * receptor_means[:, r_idx])
        )
    means = pd.DataFrame(means, index=interactions.index, columns=pair_columns)
    logger.info(f"Real geometric means time: {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    percents = np.zeros((len(counts.index), len(cluster_names)), dtype='float32')
    for i, cluster in enumerate(cluster_names):
        cells = meta[meta['cell_type'] == cluster].index
        percents[:, i] = (counts[cells] > 0).mean(axis=1).values
    percents = pd.DataFrame(percents, index=counts.index, columns=cluster_names)
    logger.info(f"Percents time: {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    shuffled_means_list = []
    for i in tqdm(range(iterations), desc="Running randomization analysis"):
        shuffled_meta = meta.copy()
        shuffled_meta['cell_type'] = np.random.permutation(shuffled_meta['cell_type'])
        
        shuffled_means = np.zeros((len(counts.index), len(cluster_names)), dtype='float32')
        for j, cluster in enumerate(cluster_names):
            cells = shuffled_meta[shuffled_meta['cell_type'] == cluster].index
            shuffled_means[:, j] = counts[cells].mean(axis=1).values
        shuffled_means = pd.DataFrame(shuffled_means, index=counts.index, columns=cluster_names)
        
        ligand_means = shuffled_means.loc[interactions['ligand']].values
        receptor_means = shuffled_means.loc[interactions['receptor']].values
        shuffled_result = np.zeros((len(interactions), len(cluster_pairs)), dtype='float32')
        for j, (s, r) in enumerate(cluster_pairs):
            s_idx, r_idx = cluster_names.index(s), cluster_names.index(r)
            shuffled_result[:, j] = np.where(
                (ligand_means[:, s_idx] == 0) | (receptor_means[:, r_idx] == 0),
                0,
                np.sqrt(ligand_means[:, s_idx] * receptor_means[:, r_idx])
            )
        shuffled_means_list.append(pd.DataFrame(shuffled_result, index=interactions.index, columns=pair_columns))
    logger.info(f"Randomization time: {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    pvalues = np.ones((len(interactions), len(cluster_pairs)), dtype='float32')
    ligand_percents = percents.loc[interactions['ligand']].values
    receptor_percents = percents.loc[interactions['receptor']].values
    
    for j, (s, r) in enumerate(cluster_pairs):
        s_idx, r_idx = cluster_names.index(s), cluster_names.index(r)
        real_means = means.iloc[:, j].values
        mask = (real_means != 0) & (ligand_percents[:, s_idx] >= threshold) & (receptor_percents[:, r_idx] >= threshold)
        if np.any(mask):
            shuffled_values = np.stack([sm.iloc[:, j].values for sm in shuffled_means_list], axis=1)
            pvalues[mask, j] = np.sum(shuffled_values[mask] > real_means[mask, None], axis=1) / iterations
    pvalues = pd.DataFrame(pvalues, index=interactions.index, columns=pair_columns)
    logger.info(f"P-values time: {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    significant_means = means.copy()
    significant_means[pvalues > pvalue_threshold] = np.nan
    logger.info(f"Significant geometric means time: {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    interactions_data = interactions[['ligand', 'receptor']].reset_index()
    pvalues = pd.concat([interactions_data, pvalues.reset_index(drop=True)], axis=1)
    means = pd.concat([interactions_data, means.reset_index(drop=True)], axis=1)
    significant_means = pd.concat([interactions_data, significant_means.reset_index(drop=True)], axis=1)
    logger.info(f"Output integration time: {time.time() - start_time:.2f} seconds")
    
    return pvalues, means, significant_means

def cpdb_exact_target(means,target_cells):
    import re
    
    t_dict=[]
    for t in target_cells:
        escaped_str = re.escape('_'+t)
        target_names=means.columns[means.columns.str.contains(escaped_str)].tolist()
        t_dict+=target_names
    target_sub=means[means.columns[:3].tolist()+t_dict]
    return target_sub

def cpdb_exact_source(means,source_cells):
    import re
    
    t_dict=[]
    for t in source_cells:
        escaped_str = re.escape(t+'_')
        source_names=means.columns[means.columns.str.contains(escaped_str)].tolist()
        t_dict+=source_names
    source_sub=means[means.columns[:3].tolist()+t_dict]
    return source_sub

def cci_interacting_heatmap(adata, 
                             celltype_key,
                             means,
                             pvalues,
                             source_cells,
                             target_cells,
                             min_means=3,
                             nodecolor_dict=None,
                             ax=None,
                             figsize=(2,6),
                             fontsize=12,
                             return_table=False):

    if nodecolor_dict is not None:
        type_color_all = nodecolor_dict
    else:
        if f'{celltype_key}_colors' in adata.uns:
            type_color_all = dict(zip(adata.obs[celltype_key].cat.categories, adata.uns[f'{celltype_key}_colors']))
        else:
            if len(adata.obs[celltype_key].cat.categories) > 28:
                type_color_all = dict(zip(adata.obs[celltype_key].cat.categories, sc.pl.palettes.default_102))
            else:
                type_color_all = dict(zip(adata.obs[celltype_key].cat.categories, sc.pl.palettes.zeileis_28))
    

    sub_means = cpdb_exact_target(means, target_cells)
    sub_means = cpdb_exact_source(sub_means, source_cells)
    

    sub_means = sub_means.loc[~sub_means['ligand'].isnull()]
    sub_means = sub_means.loc[~sub_means['receptor'].isnull()]
    
    new = sub_means.iloc[:, 3:]  
    new.index = sub_means['interaction'].tolist()
    
    cor = new.loc[new.sum(axis=1)[new.sum(axis=1) > min_means].index]
    
    sub_p = pvalues.set_index('interaction').loc[cor.index, cor.columns]
    sub_p_mat = sub_p.stack().reset_index(name="pvalue")

    corr_mat = cor.stack().reset_index(name="means")
    corr_mat['-logp'] = -np.log10(sub_p_mat['pvalue'] + 0.001)

    df_col = corr_mat['level_1'].drop_duplicates().to_frame()
    df_col['Source'] = df_col.level_1.apply(lambda x: x.split('_')[0])
    df_col['Target'] = df_col.level_1.apply(lambda x: x.split('_')[1])
    df_col.set_index('level_1', inplace=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = ax
    
    col_ha = HeatmapAnnotation(Source=anno_simple(df_col.Source,
                                                  colors=[type_color_all[i] for i in df_col.Source.unique()],
                                                  text_kws={'color': 'black', 'rotation': 0, 'fontsize': fontsize},
                                                  legend=True, add_text=False),
                               Target=anno_simple(df_col.Target,
                                                  colors=[type_color_all[i] for i in df_col.Target.unique()],
                                                  text_kws={'color': 'black', 'rotation': 0, 'fontsize': fontsize},
                                                  legend=True, add_text=False),
                               verbose=0, label_side='left',
                               label_kws={'horizontalalignment': 'right', 'fontsize': fontsize})
    
    cm = DotClustermapPlotter(corr_mat, x='level_1', y='level_0', value='means',
                              c='means', s='-logp', cmap='Reds', vmin=0,
                              top_annotation=col_ha,
                              row_dendrogram=True,
                              show_rownames=True, show_colnames=True)
    
    cm.ax_heatmap.grid(which='minor', color='gray', linestyle='--')
    
    for ax in plt.gcf().axes:
        if hasattr(ax, 'get_ylabel') and ax.get_ylabel() == 'means':
            cbar = ax
            cbar.tick_params(labelsize=fontsize)
            cbar.set_ylabel('means', fontsize=fontsize)
        ax.grid(False)
    
    ax_list = plt.gcf().axes
    ax_list[6].set_xticklabels(ax_list[6].get_xticklabels(), fontsize=fontsize)
    ax_list[6].set_yticklabels(ax_list[6].get_yticklabels(), fontsize=fontsize)
    
    if return_table:
        return cor
    else:
        return ax


def extract_interaction_edges(
    pvals: pd.DataFrame,
    alpha: float = 0.05,
    default_sep: str = "_",  # Cell type sep
    symmetrical: bool = True,
) -> pd.DataFrame:
    """exacting the significant Cell-cell interaction table: interaction_edges"""
    
    all_intr = pvals.rename(columns={"interaction": "interacting_pair"}).copy()
    intr_pairs = all_intr["interacting_pair"]
    
    col_start = 3  # interactive data starts at column 4

    all_int = all_intr.iloc[:, col_start:].T
    all_int.columns = intr_pairs

    cell_types = sorted(
        list(set([y for z in [x.split(default_sep) for x in all_intr.columns[col_start:]] for y in z]))
    )

    cell_types_comb = ["_".join(x) for x in product(cell_types, cell_types)]
    
    cell_types_keep = [ct for ct in all_int.index if ct in cell_types_comb]
    all_int = all_int.loc[cell_types_keep]

    all_count = all_int.melt(ignore_index=False).reset_index()
    
    all_count["significant"] = all_count.value < alpha
    
    count1x = all_count[["index", "significant"]].groupby("index").agg({"significant": "sum"})
    tmp = pd.DataFrame([x.split("_") for x in count1x.index])
    count_final = pd.concat([tmp, count1x.reset_index(drop=True)], axis=1)
    count_final.columns = ["SOURCE", "TARGET", "COUNT"]

    return count_final

def cci_heatmap(adata: anndata.AnnData, interaction_edges: pd.DataFrame,
                 celltype_key: str, nodecolor_dict=None, ax=None,
                 source_cells=None, target_cells=None,
                 figsize=(3, 3), fontsize=11, rotate=False, legend=True,
                 legend_kws={'fontsize': 8, 'bbox_to_anchor': (5, -0.5), 'loc': 'center left'},
                 return_table=False, **kwargs):
    
    # Color dictionary setup
    if nodecolor_dict is not None:
        type_color_all = nodecolor_dict
    else:
        if f'{celltype_key}_colors' in adata.uns:
            type_color_all = dict(zip(adata.obs[celltype_key].cat.categories, adata.uns[f'{celltype_key}_colors']))
        else:
            if len(adata.obs[celltype_key].cat.categories) > 28:
                type_color_all = dict(zip(adata.obs[celltype_key].cat.categories, sc.pl.palettes.default_102))
            else:
                type_color_all = dict(zip(adata.obs[celltype_key].cat.categories, sc.pl.palettes.zeileis_28))

    # Filter interaction edges
    corr_mat = interaction_edges.copy()
    if source_cells is not None and target_cells is None:
        corr_mat = corr_mat.loc[corr_mat['SOURCE'].isin(source_cells)]
    elif source_cells is None and target_cells is not None:
        corr_mat = corr_mat.loc[corr_mat['TARGET'].isin(target_cells)]
    elif source_cells is not None and target_cells is not None:
        corr_mat = corr_mat.loc[corr_mat['TARGET'].isin(source_cells)]
        corr_mat = corr_mat.loc[corr_mat['SOURCE'].isin(target_cells)]

    # Prepare row and column dataframes
    df_row = corr_mat['SOURCE'].drop_duplicates().to_frame()
    df_row['Celltype'] = df_row['SOURCE']
    df_row.set_index('SOURCE', inplace=True)

    df_col = corr_mat['TARGET'].drop_duplicates().to_frame()
    df_col['Celltype'] = df_col['TARGET']
    df_col.set_index('TARGET', inplace=True)

    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = ax

    # Row and column annotations
    if not rotate:
        row_ha = HeatmapAnnotation(
            TARGET=anno_simple(
                df_row.Celltype,
                colors=[type_color_all[i] for i in df_row.Celltype],
                add_text=False,
                text_kws={'color': 'black', 'rotation': 0, 'fontsize': 10},
                legend=False
            ),
            legend_gap=7,
            axis=1,
            verbose=0,
            label_kws={'rotation': 90, 'horizontalalignment': 'right', 'fontsize': 0}
        )
    else:
        row_ha = HeatmapAnnotation(
            TARGET=anno_simple(
                df_row.Celltype,
                colors=[type_color_all[i] for i in df_row.Celltype],
                add_text=False,
                text_kws={'color': 'black', 'rotation': 0, 'fontsize': 10},
                legend=False
            ),
            legend_gap=7,
            axis=0,
            verbose=0,
            label_kws={'rotation': 90, 'horizontalalignment': 'right', 'fontsize': 0}
        )

    if not rotate:
        col_ha = HeatmapAnnotation(
            SOURCE=anno_simple(
                df_col.Celltype,
                colors=[type_color_all[i] for i in df_col.Celltype],
                legend=False,
                add_text=False
            ),
            verbose=0,
            label_kws={'horizontalalignment': 'right', 'fontsize': 0},
            legend_kws={'ncols': 1},
            legend=False,
            legend_hpad=7,
            legend_vpad=5,
            axis=0
        )
    else:
        col_ha = HeatmapAnnotation(
            SOURCE=anno_simple(
                df_col.Celltype,
                colors=[type_color_all[i] for i in df_col.Celltype],
                legend=False,
                add_text=False
            ),
            verbose=0,
            label_kws={'horizontalalignment': 'right', 'fontsize': 0},
            legend_kws={'ncols': 1},
            legend=False,
            legend_hpad=7,
            legend_vpad=5,
            axis=1
        )

    import PyComplexHeatmap as pch
    if pch.__version__ > '1.7':
        hue_arg = None
    else:
        hue_arg = 'SOURCE'

    # DotClustermapPlotter
    if rotate:
        cm = DotClustermapPlotter(
            corr_mat,
            y='SOURCE',
            x='TARGET',
            value='COUNT',
            hue=hue_arg,
            legend_gap=7,
            top_annotation=col_ha,
            left_annotation=row_ha,
            c='COUNT',
            s='COUNT',
            cmap='Reds',
            vmin=0,
            show_rownames=False,
            show_colnames=False,
            row_dendrogram=False,
            col_names_side='left',
            legend=legend,
            **kwargs
        )
    else:
        cm = DotClustermapPlotter(
            corr_mat,
            x='SOURCE',
            y='TARGET',
            value='COUNT',
            hue=hue_arg,
            legend_gap=7,
            top_annotation=row_ha,
            left_annotation=col_ha,
            c='COUNT',
            s='COUNT',
            cmap='Reds',
            vmin=0,
            show_rownames=False,
            show_colnames=False,
            row_dendrogram=False,
            col_names_side='top',
            legend=legend,
            **kwargs
        )

    cm.ax_heatmap.grid(which='minor', color='gray', linestyle='--', alpha=0.5)
    cm.ax_heatmap.grid(which='major', color='black', linestyle='-', linewidth=0.5)
    cm.cmap_legend_kws = {'ncols': 1}

    # Adjust axes labels
    if not rotate:
        for ax in plt.gcf().axes:
            if hasattr(ax, 'get_ylabel'):
                if ax.get_ylabel() == 'COUNT':
                    cbar = ax
                    cbar.tick_params(labelsize=fontsize)
                    cbar.set_ylabel('COUNT', fontsize=fontsize)
                if ax.get_xlabel() == 'SOURCE':
                    ax.xaxis.set_label_position('top')
                    ax.set_ylabel('Target', fontsize=fontsize)
                if ax.get_ylabel() == 'TARGET':
                    ax.xaxis.set_label_position('top')
                    ax.set_xlabel('Source', fontsize=fontsize)
            ax.grid(False)
    else:
        for ax in plt.gcf().axes:
            if hasattr(ax, 'get_ylabel'):
                if ax.get_ylabel() == 'COUNT':
                    cbar = ax
                    cbar.tick_params(labelsize=fontsize)
                    cbar.set_ylabel('COUNT', fontsize=fontsize)
                if ax.get_ylabel() == 'SOURCE':
                    ax.xaxis.set_label_position('top')
                    ax.set_xlabel('Target', fontsize=fontsize)
                if ax.get_xlabel() == 'TARGET':
                    ax.xaxis.set_label_position('top')
                    ax.set_ylabel('Source', fontsize=fontsize)
            ax.grid(False)

    # Legend setup
    handles = [plt.Line2D([0], [0], color=type_color_all[cell], lw=4) for cell in type_color_all.keys()]
    labels = type_color_all.keys()

    # Place legend without causing layout issues
    if legend:
        plt.legend(handles, labels, 
                   borderaxespad=1, handletextpad=0.5, labelspacing=0.2, **legend_kws)

    plt.subplots_adjust(left=0.1, right=0.9, top=1.0, bottom=0.1)  # Adjust these values as needed

    if return_table:
        return corr_mat
    else:
        return ax

def cci_chord(adata:anndata.AnnData,interaction_edges:pd.DataFrame,
                      celltype_key:str,count_min=50,nodecolor_dict=None,
                      fontsize=12,padding=80,radius=100,save='chord.svg',
                      rotation=0,bg_color = "#ffffff",bg_transparancy = 1.0):
    import itertools
    import openchord as ocd
    data=interaction_edges.loc[interaction_edges['COUNT']>count_min].iloc[:,:2]
    data = list(itertools.chain.from_iterable((i, i[::-1]) for i in data.values))
    matrix = pd.pivot_table(
        pd.DataFrame(data), index=0, columns=1, aggfunc="size", fill_value=0
    ).values.tolist()
    unique_names = sorted(set(itertools.chain.from_iterable(data)))

    matrix_df = pd.DataFrame(matrix, index=unique_names, columns=unique_names)

    if nodecolor_dict!=None:
        type_color_all=nodecolor_dict
    else:
        if '{}_colors'.format(celltype_key) in adata.uns:
            type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,adata.uns['{}_colors'.format(celltype_key)]))
        else:
            if len(adata.obs[celltype_key].cat.categories)>28:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.default_102))
            else:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.zeileis_28))
    
    fig=ocd.Chord(matrix, unique_names,radius=radius)
    fig.colormap=[type_color_all[u] for u in unique_names]
    fig.font_size=fontsize
    fig.padding = padding
    fig.rotation = rotation
    fig.bg_color = bg_color
    fig.bg_transparancy = bg_transparancy
    if save!=None:
        fig.save_svg(save)
    return fig

def cci_network(adata:anndata.AnnData,interaction_edges:pd.DataFrame,
                      celltype_key:str,nodecolor_dict=None,counts_min=50,
                       source_cells=None,target_cells=None,
                      edgeswidth_scale:int=1,nodesize_scale:int=1,
                      figsize:tuple=(4,4),title:str='',
                      fontsize:int=12,ax=None,
                     return_graph:bool=False):
    G=nx.DiGraph()
    for i in interaction_edges.index:
        if interaction_edges.loc[i,'COUNT']>counts_min:
            G.add_edge(interaction_edges.loc[i,'SOURCE'],
                       interaction_edges.loc[i,'TARGET'],
                       weight=interaction_edges.loc[i,'COUNT'],)
        else:
            G.add_edge(interaction_edges.loc[i,'SOURCE'],
                       interaction_edges.loc[i,'TARGET'],
                       weight=0,)

    if nodecolor_dict!=None:
        type_color_all=nodecolor_dict
    else:
        if '{}_colors'.format(celltype_key) in adata.uns:
            type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,adata.uns['{}_colors'.format(celltype_key)]))
        else:
            if len(adata.obs[celltype_key].cat.categories)>28:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.default_102))
            else:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.zeileis_28))

    G_nodes_dict={}
    links = []
    for i in G.edges:
        if i[0] not in G_nodes_dict.keys():
            G_nodes_dict[i[0]]=0
        if i[1] not in G_nodes_dict.keys():
            G_nodes_dict[i[1]]=0
        links.append({"source": i[0], "target": i[1]})
        weight=G.get_edge_data(i[0],i[1])['weight']
        G_nodes_dict[i[0]]+=weight
        G_nodes_dict[i[1]]+=weight

    edge_li=[]
    for u,v in G.edges:
        if G.get_edge_data(u, v)['weight']>0:
            if source_cells==None and target_cells==None:
                edge_li.append((u,v))
            elif source_cells!=None and target_cells==None:
                if u in source_cells:
                    edge_li.append((u,v))
            elif source_cells==None and target_cells!=None:
                if v in target_cells:
                    edge_li.append((u,v))
            else:
                if u in source_cells and v in target_cells:
                    edge_li.append((u,v))


    import matplotlib.pyplot as plt
    import numpy as np
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize) 
    else:
        ax=ax
    pos = nx.circular_layout(G)
    p=dict(G.nodes)
    
    nodesize=np.array([G_nodes_dict[u] for u in G.nodes()])/nodesize_scale
    nodecolos=[type_color_all[u] for u in G.nodes()]
    nx.draw_networkx_nodes(G, pos, nodelist=p,node_size=nodesize,node_color=nodecolos)
    
    edgewidth = np.array([G.get_edge_data(u, v)['weight'] for u, v in edge_li])
    edgewidth=np.log10(edgewidth+1)/edgeswidth_scale
    edgecolos=[type_color_all[u] for u,o in edge_li]
    nx.draw_networkx_edges(G, pos,width=edgewidth,edge_color=edgecolos,edgelist=edge_li)
    plt.grid(False)
    plt.axis("off")
    
    pos1=dict()
    for i in G.nodes:
        pos1[i]=pos[i]
    from adjustText import adjust_text
    import adjustText
    from matplotlib import patheffects
    texts=[ax.text(pos1[i][0], 
               pos1[i][1],
               i,
               fontdict={'size':fontsize,'weight':'normal','color':'black'},
                path_effects=[patheffects.withStroke(linewidth=2, foreground='w')]
               ) for i in G.nodes if 'ENSG' not in i]
    if adjustText.__version__<='0.8':
        adjust_text(texts,only_move={'text': 'xy'},arrowprops=dict(arrowstyle='->', color='red'),)
    else:
        adjust_text(texts,only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
                    arrowprops=dict(arrowstyle='->', color='black'))
        
    plt.title(title,fontsize=fontsize+1)

    if return_graph==True:
        return G
    else:
        return ax

def cci_interacting_network(adata,
                             celltype_key,
                             means,
                             source_cells,
                             target_cells,
                             means_min=0,
                             means_sum_min=0,        
                             nodecolor_dict=None,
                             ax=None,
                             figsize=(6,6),
                             fontsize=10,
                             return_graph=False):
    """
    Creates and visualizes a network of cell-cell interactions.

    Parameters:
    adata : AnnData
        AnnData object containing cell type and associated data.
    celltype_key : str
        Column name for cell types.
    means : DataFrame
        DataFrame containing interaction strengths.
    source_cells : list
        List of source cell types.
    target_cells : list
        List of target cell types.
    means_sum_min : float, optional
        Minimum threshold for interaction strength of ligand-receptor (default is 0).
    means_min : float, optional
        Minimum threshold for the sum of individual interactions (default is 0).
    nodecolor_dict : dict, optional
        Dictionary mapping cell types to colors (default is None).
    ax : matplotlib.axes.Axes, optional
        Axes object for the plot (default is None).
    figsize : tuple, optional
        Size of the figure (default is (6, 6)).
    fontsize : int, optional
        Font size for node labels (default is 10).



    Returns:
    ax : matplotlib.axes.Axes
        Axes object with the drawn network.
    """
    from adjustText import adjust_text
    import re
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # Determine node colors
    if nodecolor_dict:
        type_color_all = nodecolor_dict
    else:
        color_key = f"{celltype_key}_colors"
        categories = adata.obs[celltype_key].cat.categories
        if color_key in adata.uns:
            type_color_all = dict(zip(categories, adata.uns[color_key]))
        else:
            palette = sc.pl.palettes.default_102 if len(categories) > 28 else sc.pl.palettes.zeileis_28
            type_color_all = dict(zip(categories, palette))

    # Create a directed graph
    G = nx.DiGraph()

    # Filter the means DataFrame
    sub_means = cpdb_exact_target(means, target_cells)
    sub_means = cpdb_exact_source(sub_means, source_cells)
    sub_means = sub_means.loc[~sub_means['ligand'].isnull()]
    sub_means = sub_means.loc[~sub_means['receptor'].isnull()]

    # Build the graph
    nx_dict = {}
    for source_cell in source_cells:
        for target_cell in target_cells:
            key = f"{source_cell}_{target_cell}"
            nx_dict[key] = []
            escaped_str = re.escape(key)
            receptor_names = sub_means.columns[sub_means.columns.str.contains(escaped_str)].tolist()
            receptor_sub = sub_means[sub_means.columns[:3].tolist() + receptor_names]

            for j in receptor_sub.index:
                if receptor_sub.loc[j, receptor_names].sum() > means_sum_min:
                    for rece in receptor_names:
                        if receptor_sub.loc[j, rece] > means_min:
                            nx_dict[key].append(receptor_sub.loc[j, 'receptor'])
                            G.add_edge(source_cell, f'L:{receptor_sub.loc[j, "ligand"]}')
                            G.add_edge(f'L:{receptor_sub.loc[j, "ligand"]}', f'R:{receptor_sub.loc[j, "receptor"]}')
                            G.add_edge(f'R:{receptor_sub.loc[j, "receptor"]}', rece.split('_')[1])
            nx_dict[key] = list(set(nx_dict[key]))

    # means_sum_min:
    # If the sum of interaction strengths of a given ligand-receptor pair 
    # across all source-target cell pairs is greater than this threshold, 
    # then this ligand-receptor pair is considered for inclusion in the network.
    # (i.e., filter out globally weak ligand-receptor pairs)
    
    # means_min:
    # For a ligand-receptor pair that passes the global threshold (means_min),
    # only those specific cell-cell interactions (CCI) where the interaction 
    # strength exceeds this threshold will be shown in the network.
    # (i.e., filter out weak CCIs even if the ligand-receptor pair is overall strong)



    # Set colors for ligand and receptor nodes
    color_dict = type_color_all
    color_dict['ligand'] = '#a51616'  # Red for ligands
    color_dict['receptor'] = '#c2c2c2'  # Gray for receptors

    # Assign colors to nodes
    node_colors = [
        color_dict.get(node, 
                       color_dict['ligand'] if 'L:' in node 
                       else color_dict['receptor'])
        for node in G.nodes()
    ]

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Define shells for shell_layout
    source_nodes = [n for n in G.nodes() if n in source_cells]
    ligand_nodes = [n for n in G.nodes() if 'L:' in n]
    receptor_nodes = [n for n in G.nodes() if 'R:' in n]
    target_nodes = [n for n in G.nodes() if n in target_cells and n not in source_cells]
    shells = [source_nodes, ligand_nodes, receptor_nodes, target_nodes]

    # Use shell_layout
    pos = nx.shell_layout(G, nlist=shells)

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='#c2c2c2', ax=ax)

    # Add labels to the nodes
    texts = [
        ax.text(pos[node][0], pos[node][1], node,
                fontdict={'size': fontsize, 'weight': 'bold', 'color': 'black'})
        for node in G.nodes() if 'ENSG' not in node
    ]
    adjust_text(texts, only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
                arrowprops=dict(arrowstyle='-', color='black'))

    # Remove axes
    ax.axis("off")

    if return_graph:
        return G
    else:
        return ax
