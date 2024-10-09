# DDPM
This project is completely inspired by "Synthetic ECG Signal Generation using Probabilistic Diffusion Models.pdf", the images from this file were all extracted from it. 
Initial, use Denoising Diffusion Probabilistic Model with U-Net architect to synthesize the ECG data from mitdb dataset
The pipiline is below:
![image](https://github.com/user-attachments/assets/bed6e6e1-78bb-4b62-88a4-7c04a23a1bc3)
Firstly, we need to convert ECG signal from Cartesian Coordinate into Polar Coordinate Coordinate 
![image](https://github.com/user-attachments/assets/f2645a34-d620-40a1-823c-d681935efbbb)

Then we used the polarized ECG signal to 3 distinct fields: GASF( Grammian Angular Summation Field), GADF( Grammian Angular Difference Field) and MTF(Markov Transition Field) and concatenate them as an image

From the image we used DDPM method to train the Unet and extract the signal from the synthesize image. 
