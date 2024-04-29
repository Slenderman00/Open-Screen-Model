wget -O dataset.zip https://huggingface.co/datasets/goldpulpy/Image-human-mask/resolve/main/dataset.zip?download=true
unzip dataset.zip
rm dataset.zip

kaggle datasets download -d tapakah68/supervisely-filtered-segmentation-person-dataset
unzip supervisely-filtered-segmentation-person-dataset.zip
rm rf.csv
rm supervisely-filtered-segmentation-person-dataset.zip

cd supervisely_person_clean_2667_img/supervisely_person_clean_2667_img/images
for file in *.png; do
    ffmpeg -i "$file" -q:v 2 "${file%.png}.jpg"
    rm "$file"
done