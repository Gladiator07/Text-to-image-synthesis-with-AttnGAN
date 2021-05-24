# Setup script for Google Cloud VM
echo "Installing dependencies"
pip3 install six
pip3 install black

cd /content/Text-to-image-synthesis-with-GANs/
cp -r /content/drive/MyDrive/Project_Data/Text-to-image-AttnGAN/data .

cd data/
tar -xf CUB_200_2011.tgz
tar -xf text.tar.xz

rm CUB_200_2011.tgz
rm text.tar.xz

echo "Workspace setup done ..."