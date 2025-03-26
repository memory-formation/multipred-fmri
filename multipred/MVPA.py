import nibabel as nib
import numpy as np
from nilearn.input_data import NiftiMasker
from sklearn.svm import SVC


def get_main_inds(df, modality, decoded_stim = "presented"):
    
    if decoded_stim == "presented":
        if modality == "visual":
            target_stims = ["45", "135"] 
            alternative_stims = ["100", "160"]
            target_stim_col = "v_trailing" 
            target_pred_col = "v_pred"
            alternative_stim_col = "a_trailing"
            alternative_pred_col = "a_pred"
        elif modality == "auditory":
            target_stims = ["100", "160"]
            alternative_stims = ["45", "135"]
            target_stim_col = "a_trailing"
            target_pred_col = "a_pred"
            alternative_stim_col = "v_trailing"
            alternative_pred_col = "v_pred"
            
    elif decoded_stim == "predicted":
        if modality == "visual":
            target_stims = ["45", "135"] 
            alternative_stims = ["100", "160"]
            target_stim_col = "expected_v" 
            target_pred_col = "v_pred"
            alternative_stim_col = "expected_a"
            alternative_pred_col = "a_pred"
        elif modality == "auditory":
            target_stims = ["100", "160"]
            alternative_stims = ["45", "135"]
            target_stim_col = "expected_a"
            target_pred_col = "a_pred"
            alternative_stim_col = "expected_v"
            alternative_pred_col = "v_pred"
            
    
    trial_inds = {f"EXP_{target_stims[0]}": [], 
                  f"EXP_{target_stims[1]}": [], 
                  f"VP_{target_stims[0]}": [], 
                  f"VP_{target_stims[1]}": []} # dictionary to store indices of trial subsets grouped by conditions

    for target_pred in ["EXP", "VP"]:
        for target_stim in target_stims:
            if target_pred == "VP": 
                # we can get all indices 
                inds = df[(df[target_pred_col] == target_pred) & (df[target_stim_col] == target_stim)].index
                rand_inds = np.random.choice(inds, 8, replace=False) # get 8 random indices without replacement. Making for a total of 8 for each target For VP
                trial_inds[f"{target_pred}_{target_stim}"] = rand_inds 
            else: # If not we need to reduce number of trials to 8
                # We will do so while balancing the number alternative_stim and alternative_pred
                for alternative_pred in ["EXP", "VP"]:
                    for alternative_stim in alternative_stims:
                        inds = df[(df[target_pred_col] == target_pred) & (df[target_stim_col] == target_stim) & (df[alternative_pred_col] == alternative_pred) & (df[alternative_stim_col] == alternative_stim)].index
                        # additionally, we should make sure that the 2 selected indices are not already present in the corresponding entry of trial_inds dictionary
                        while True:
                            rand_inds = np.random.choice(inds, 2, replace=False)
                            if not any([ind in trial_inds[f"{target_pred}_{target_stim}"] for ind in rand_inds]):
                                break
                        trial_inds[f"{target_pred}_{target_stim}"] += list(rand_inds)

    return trial_inds


def main_stimuli_labels(main_df, subj, modality, decoded_stim="presented"):
    if main_df[main_df['id'] == subj]["start_mod"].values[0] == 0: main_blocks = [0, 1, 2, 3, 4, 5]
    else: main_blocks = [1, 0, 3, 2, 5, 4]

    if decoded_stim == "presented":
        if modality == "visual": stim = "v_trailing"
        elif modality == "auditory": stim = "a_trailing"
        else: raise ValueError("modality must be either 'visual' or 'auditory'")
    elif decoded_stim == "predicted":
        if modality == "visual": stim = "expected_v"
        elif modality == "auditory": stim = "expected_a"
        else: raise ValueError("modality must be either 'visual' or 'auditory'")
    
    # Get stimuli labels at each trial of every run from behavioral data
    main1_labels = main_df[(main_df['id'] == subj) & (main_df['block']==main_blocks[0]) & (main_df['catch']==0)][stim].values
    main2_labels = main_df[(main_df['id'] == subj) & (main_df['block']==main_blocks[1]) & (main_df['catch']==0)][stim].values
    main3_labels = main_df[(main_df['id'] == subj) & (main_df['block']==main_blocks[2]) & (main_df['catch']==0)][stim].values
    main4_labels = main_df[(main_df['id'] == subj) & (main_df['block']==main_blocks[3]) & (main_df['catch']==0)][stim].values
    main5_labels = main_df[(main_df['id'] == subj) & (main_df['block']==main_blocks[4]) & (main_df['catch']==0)][stim].values
    main6_labels = main_df[(main_df['id'] == subj) & (main_df['block']==main_blocks[5]) & (main_df['catch']==0)][stim].values

    return main1_labels, main2_labels, main3_labels, main4_labels, main5_labels, main6_labels


def load_func(path, mask, clip = False):
    niimg = nib.load(path)
    niimg_data = niimg.get_fdata()
    masker = NiftiMasker(mask_img=mask, standardize=True)
    mask_data = mask.get_fdata()

    if clip:
        a,b = np.percentile(niimg_data[mask_data == 1].ravel(), [0.1, 99.9])
        np.clip(niimg_data, a, b, out=niimg_data)

        niimg = nib.Nifti1Image(niimg_data, niimg.affine, niimg.header)

    # apply mask to data
    masked_data = masker.fit_transform(niimg)
    niimg = masker.inverse_transform(masked_data)

    return niimg


def get_mean_data(niimg, inds):
    """ Get mean data for each condition from a 4D nifti image with all trials of a run"""
    niimg_data = niimg.get_fdata() # get data from nifti image
    # loop through the conditions and get mean data for each condition, then store them in a 4d array
    avg_data = np.empty((niimg_data.shape[0], niimg_data.shape[1], niimg_data.shape[2], len(inds)))
    for cond in inds:
        cond_data = niimg_data[..., inds[cond]].mean(axis=3) # slice niimg using the indices from inds dictionary and get mean data for each condition
        avg_data[..., list(inds.keys()).index(cond)] = cond_data
    
    # trnsform the 4d array to a nifti image 
    avg_niimg = nib.Nifti1Image(avg_data, niimg.affine, niimg.header)
    return avg_niimg


def stratify_data(niimg, inds):
    
    niimg_data = niimg.get_fdata() # get data from nifti image
    # loop through the conditions and select volumes based on inds, then store them in a 4d array
    stratified_data = np.empty((niimg_data.shape[0], niimg_data.shape[1], niimg_data.shape[2], 0)) # create an empty 4d array to store mean data for each condition
    stratified_labels = []
    for cond in inds:
        cond_data = niimg_data[..., inds[cond]] # slice niimg using the indices from inds dictionary and get mean data for each condition
        stratified_data = np.concatenate((stratified_data, cond_data), axis=3)

    return stratified_data


def train_test_model(X_train, X_test, y_train, c=1):
    """ 
    Train and test a model using a given training and testing data 
    """

    svc_model = SVC(kernel='linear', C= c, probability=True)  

    # Train the model
    svc_model.fit(X_train, y_train)

    # Make predictions on the test dataset
    test_predictions = svc_model.predict(X_test)
    
    #test probabilities
    test_probabilities = svc_model.predict_proba(X_test)

    # Get decision function values
    decision_distances = svc_model.decision_function(X_test)

    return test_predictions, test_probabilities, decision_distances 
