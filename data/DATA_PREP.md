# Data Preparation

1.  Download the VidSitu Annotations from here: [Onedrive link annotations](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/zeeshan_khan_research_iiit_ac_in/EVgGBdurnitAlPwIsf4dlvABXmlVSZhFTMa-a5LSuE7ibA?e=SiMvhB)

    The directory should look as follows:

    ```
    data
    └── vidsitu_annotations
        ├── split_files
        │   ├── vseg_split_testevrel_lb.json
        │   ├── vseg_split_testsrl_lb.json
        │   ├── vseg_split_testvb_lb.json
        │   ├── vseg_split_train_lb_new.json
        │   └── vseg_split_valid_lb_new.json
        ├── vinfo_files
        │   ├── vinfo_train_lb.json
        │   └── vinfo_valid_lb.json
        └── vseg_ann_files
            ├── vsann_testevrel_noann_lb.json
            ├── vsann_testsrl_noann_lb.json
            ├── vsann_train_lb.json
            ├── vsann_valid_lb.json
            └── GT_grounding.json
    ```

2. Download the vocabulary files from here: [google drive link vocab](https://drive.google.com/file/d/1TAreioObLGKqU7M9wmnuaXh4b5s_2YdK/view?usp=sharing) 
and place them under `data/vsitu_vocab`
    ```
    function gdrive_download () {
        CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
        wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
        rm -rf /tmp/cookies.txt
    }

    cd $ROOT/data
    export VOCAB_ZIP_DRIVE_ID="1TAreioObLGKqU7M9wmnuaXh4b5s_2YdK" # to be filled after upload
    gdrive_download $VOCAB_ZIP_DRIVE_ID vsitu_vocab.zip
    unzip vsitu_vocab.zip -d vsitu_vocab
    rm vsitu_vocab.zip
    ```

3. Download all the subsampled frames from here: [One drive link subsampled frames](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/zeeshan_khan_research_iiit_ac_in/ETC5KbWNsGVMmutDluGfXsoBBkNyPoUJLJHnpa5AOaVLvw?e=tlwFrH)
    Place them in `data/vsitu_11_frames_per_vid`

4.  directly download the pre-extracted video features from [google drive link video features](https://drive.google.com/file/d/1rBrRmew7Soul51MjLN6F55oTEzUfzyXv/view)

    To download directly on the remote, you can use the following convenience function

    ```
    function gdrive_download () {
        CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
        wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
        rm -rf /tmp/cookies.txt
    }

    cd $ROOT/data
    export FEATURE_ZIP_DRIVE_ID="1rBrRmew7Soul51MjLN6F55oTEzUfzyXv" # to be filled after upload
    gdrive_download "1rBrRmew7Soul51MjLN6F55oTEzUfzyXv" vsitu_vidfeats_drive.zip
    unzip vsitu_vidfeats_drive.zip -d vsitu_vid_feats
    rm vsitu_vidfeats_drive.zip
    ```
    GVSR uses the Slowfast features, place them in `features/vsitu_vid_feats`.

5. Download the pre-extracted object features from here: [One drive link object features](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/lakshmipathi_balaji_research_iiit_ac_in/ETGCnb58rORJnZuyIJ-dLjYBIKx1PExkx5etvr2exBcHOA?e=YoELTZ&download=1)
    Object features are provided for the top 15 objects from all the 11 frames that are subsapled at 1 FPS from each video.

    Alternatively you can setup the Bottom-up-top-down attention's Faster RCNN provided by https://github.com/airsplay/py-bottom-up-attention and extract the object features for the subsampled frames.

    Place the object featurs in `features/vsitu_all_11_frames_feats`



