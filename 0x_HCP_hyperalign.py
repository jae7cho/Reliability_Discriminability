#############################################################################
#############################################################################
# 1-step script to create hyperaligned HCP data in 10k:                     #
# * Using rfMRI_REST1 LR and RL projection matrices as 'training'           #
#   for hyperalignment projections                                          #
# * Apply projection matrices to all other tasks ('testing')                # 
# Steps:                                                                    #
# 1) Create connectivity matrices using concatenated rfMRI_REST1 LR and RL  #
# 2) Hyperalign connectivity matrices to Feilong 80 10k template            #
# 3) Apply projection matrices to all other tasks except rfMRI_REST1        #
#############################################################################
#############################################################################

import os,sys
import numpy as np
import nibabel as nib
sys.path.append('/home/jcho/hypertools_edit')
from correlation import correlation as anicor
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
from mvpa2.datasets import Dataset
from mvpa2.datasets.gifti import gifti_dataset
from mvpa2.mappers.zscore import zscore, ZScoreMapper
from mvpa2.base.hdf5 import h5load,h5save
from mvpa2.support.nibabel import surf
from mvpa2.mappers.procrustean import ProcrusteanMapper
from mvpa2.misc.surfing.queryengine import SurfaceQueryEngine
from myprocrustes1 import myprocrustes as myp
from itertools import product
import cifti
import time

# Input is just subject name:
sub = sys.argv[1]

ndir = ""
#ndir = "/Users/jaewook.cho/Desktop/mnt"
tempdir = '%s/data3/cdb/jcho/hcp180' % ndir

def hypalign(source, target, node, surfsel, projmat,mask):
    slneighbors = surfsel[node]
    if len(slneighbors) == 0:
        return projmat

    sl = []
    sl = [source[:, slneighbors], target[:, slneighbors]]
    
    hmapper = ProcrusteanMapper(space='commonspace')
    refds = sl[1].copy()
    commonspace = refds.samples
    zscore(commonspace,chunks_attr=None)
    ds_new = sl[0].copy()
    ds_new.sa[hmapper.get_space()] = commonspace.astype(float)
    hmapper.train(ds_new)
    conproj = hmapper.proj
    m, n = conproj.shape
    index = np.array(slneighbors)
    projmat[np.ix_(index,index)] += np.asarray(conproj)
    return projmat

################################################################
################################################################

#######################
# Create Connectomes: #
#######################

# Define structure name per hemisphere to extract from cifti data:
brainstruct = {'L':'CIFTI_STRUCTURE_CORTEX_LEFT','R':'CIFTI_STRUCTURE_CORTEX_RIGHT'}
# Load connectivity targets and mask:
targimg = {}
targts = []
maskimg = {}
targets = {}
subimg = {}
subds = []
zdss = []
ind = {}
for ses in ['LR','RL']:
    cname = '%s/%s/rfMRI_REST1_%s/rfMRI_REST1_%s_10k_Atlas_MSMAll_clean_24nuisance.reg_lin.trend_filt_sm6.dtseries.nii' % (tempdir,sub,ses,ses)
    tmpimg =cifti.read(cname)
    zds = tmpimg[0]
    zscore(zds,chunks_attr=None)
    nverts = zds.shape[1]
    zdss.append(zds)
    for hemi in ['L', 'R']:
        # Mask:
        tmpmask = gifti_dataset(
            '%s/outgoing/%s.HCP_10k_mask.func.gii' % (tempdir,hemi))
        maskimg[hemi] = np.where(tmpmask.samples != 0)[1]
        # Connectivity Targets:
        tmptargs = gifti_dataset(
            '%s/resources/700_data/%s.700_to_10k_conn_targs.func.gii' % (tempdir, hemi))
        tmptargs = tmptargs[:, maskimg[hemi]]
        targimg[hemi] = np.where(tmptargs.samples != 0)[1]
        # Extract hemi data:
        ind[hemi] = [index for index in range(nverts) if tmpimg[1][1][index][2] == '%s' % brainstruct[hemi]]
        subimg[hemi] = zds[:,ind[hemi]]
        # Mask data and get connectivity target timeseries
        tmp = subimg[hemi][:, maskimg[hemi]]
        targets[hemi] = tmp[:, targimg[hemi]]
    # Combine left and right target timeseries
    targts.append(np.c_[targets['L'], targets['R']])
    subds.append(np.c_[subimg['L'], subimg['R']])

# Concatenate LR and RL data and targets:
ds = np.vstack(subds)
targts = np.vstack(targts)
# Create connectivity matrix:
corr = anicor(targts.T, ds.T)

###############
# Hyperalign: #
###############

# Load and set up template and searchlights:
target = {}
searchlights = {}
maskimg = {}
for hemi in ['L','R']:
    print 'Loading %s mask' % hemi
    tmpmask = gifti_dataset('%s/outgoing/%s.HCP_10k_mask.func.gii' % (tempdir,hemi))
    maskimg[hemi] = np.where(tmpmask.samples!=0)[1]
    # Load template
    print 'Loading and zscore %s hemisphere template' % hemi
    tt = time.time()
    target[hemi] = h5load('%s/template/jSL_cspace_%s.gzipped.hdf5'  % (tempdir, hemi))
    zscore(target[hemi],chunks_attr=None)
    ttt = time.time()
    tttt = ttt-tt
    print 'Loading and zscore %s hemisphere template took %s seconds' % (hemi, tttt)

# Run Hyperalignment per hemisphere:
hypsource = {}
finalproj = {}
for hemi in ['L','R']:
    # Load connectvity profile:
    t1 = time.time()
    ds = corr[:,ind[hemi]]
    source = Dataset(ds)
    zscore(source,chunks_attr=None)
    # Set 'node_indices' for hyperalignment function:
    source.fa['node_indices'] = range(len(ind[hemi]))
    source = source[:,maskimg[hemi]]
    # Read surface for searchlight vertices:
    s = surf.read('%s/resources/10k/S1200.%s.midthickness.10k_fs_LR.surf.gii' % (tempdir,hemi))
    surfsel = SurfaceQueryEngine(surface=s,radius=12,distance_metric='dijkstra',fa_node_key='node_indices')
    surfsel.train(source)
    nfeatures = source.shape[1]
    # Searchlight hyperalignment on connectomes:
    projmat = np.zeros([nfeatures,nfeatures])
    [hypalign(source,target[hemi],node,surfsel,projmat,maskimg[hemi]) for node in maskimg[hemi]]
    normproj = projmat.copy()
    # Normalise projection matrices to keep scale of values:
    for ii in range(len(projmat)):
        nnz = np.count_nonzero(normproj[:, ii])
        nnz = np.float(nnz)
        normproj[:, ii] = normproj[:, ii]/nnz
    normproj = np.nan_to_num(normproj)
    dd = np.asmatrix(source.samples)
    dd = dd - np.mean(dd,axis=0)
    next = (dd * normproj).A
    hypsource[hemi] = next
    zscore(hypsource[hemi],chunks_attr=None)
    finalproj[hemi] = normproj
    t2 = time.time()
    print('Hyperalignment for %s hemisphere took %s minutes' % (hemi, ((t2-t1)/60.)))

# Apply projections to other tasks:
# tasks = ['rfMRI_REST2','tfMRI_WM','tfMRI_GAMBLING','tfMRI_LANGUAGE','tfMRI_MOTOR','tfMRI_RELATIONAL','tfMRI_SOCIAL']
tasks = ['tfMRI_RELATIONAL']
for task in tasks:
    for ses in ['LR','RL']:
        if task == 'rfMRI_REST2':
            cname = '%s/%s/%s_%s/%s_%s_10k_Atlas_MSMAll_clean_24nuisance.reg_lin.trend_filt_sm6' % (tempdir,sub,task,ses,task,ses)
        else:
            cname = '%s/%s/%s_%s/%s_%s_Atlas_MSMAll_10k_anat_cortex' % (tempdir,sub,task,ses,task,ses)
        
        tmpimg =cifti.read('%s.dtseries.nii' % cname)
        tmpwrite = tmpimg[0].copy()
        nvols = tmpimg[0].shape[0]
        for hemi in ['L','R']:
            nverts = len(ind[hemi])
            blank = np.zeros([nvols,nverts])
            img = tmpimg[0][:,ind[hemi]]
            img = img[:,maskimg[hemi]]
            himg1 = img.dot(finalproj[hemi])
            blank[:,maskimg[hemi]] = himg1
            tmpwrite[:,ind[hemi]] = blank
        cifti.write('%s/%s/%s_%s/%s_%s_Atlas_MSMAll_10k_R1LRRL_hyp.dtseries.nii' % (tempdir,sub,task,ses,task,ses),tmpwrite,tmpimg[1])
