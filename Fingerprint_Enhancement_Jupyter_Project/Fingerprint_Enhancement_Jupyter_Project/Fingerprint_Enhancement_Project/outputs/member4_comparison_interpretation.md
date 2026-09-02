# Four-Member Comparison Interpretation

## Member 1: Non-Local Means Denoising

NLM preserved the 29+ dB behaviour on the development split: mean PSNR 29.4366 dB, MSE 0.00114977, SSIM 0.9408. It remains above the 28.17 dB target. On direct Altered images, its mean coherence was 0.7180, fragmentation 2.5610, and contrast delta versus P0 -0.0050. Main weakness: it can slightly soften local contrast even while improving denoising.

## Member 2: Modified / Orientation-Adaptive Gabor

Modified Gabor produced controlled PSNR 25.0002 dB and SSIM 0.8373. On direct Altered images, coherence was 0.6863, contrast delta versus P0 -0.0153, and fragmentation delta versus P0 0.2954. Compared with Ordinary Gabor controlled PSNR 18.0003 dB, the adaptive method is scientifically deeper and usually less extreme, but it still changes ridge intensities and must be used carefully.

## Member 3: Total Variation Restoration

TV produced controlled PSNR 29.1850 dB, MSE 0.00121547, and SSIM 0.9402. It is a strong denoising/restoration method under Gaussian noise, but this fixed setting should be compared against NLM rather than assumed superior. On direct Altered images, TV coherence was 0.6927, fragmentation 2.7552, and contrast delta versus P0 -0.0089.

## Member 4: Coherence-Guided Directional Diffusion

Directional diffusion produced controlled PSNR 27.8028 dB and SSIM 0.8684. On direct Altered images, coherence was 0.6780, fragmentation 2.7440, and contrast delta versus P0 -0.0147. Its benefit is structural and orientation-aware; its risk is PSNR loss if it smooths ridge detail after NLM.

## Comparator Notes

Ordinary Gabor remains a baseline, not the final M2 contribution. In this run its Altered coherence was 0.7353, contrast delta versus P0 -0.0494, and fragmentation delta versus P0 1.3578. Wiener controlled PSNR was 28.4338 dB. Gaussian filtering was retained as a historical baseline in the registry; this small comparison used the optional baselines requested here.

## Current Individual Winner

The strongest individual controlled member technique is Non-Local Means Denoising with PSNR 29.4366 dB.
