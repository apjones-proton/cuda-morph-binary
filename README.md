# Efficient Morphology Operations for Binary GPU Images

This library performs morphology operations (dilate, erode, open, close) 
on binary images already stored in CUDA GPU memory.
It is efficient even for large dilations because it first packs scanlines into 64 bit words,
1 bit per pixel, and dilates up to 64 pixels simultaneously on every thread
using fast bit-manipulation operation 
and updates up to 64 pixels in every write operation. 
It also skips pixels that can be inferred to have null effect.

I have included some basic tests and benchmarks against a small selection of artificial 
cases. 

The tests compare for pixel-perfect agreement with simple 
'obviously correct' cpu implementation as a reference 
(although I optimized it so that I could run larger sizes to confirm they are 
correct so it is not the simplest possible and thus less obviously 
correct than it could be). 

The benchmarks compare against that CPU version and against NVIDIA Performance 
Primitives 8-bit dilation because there is no 1-bit dilation and this is the best 
NPP option available. This library is orders of magnitude faster than both in most cases.
The NPP function struggles with these large dilations, mostly performing even worse 
than a single cpu, probably because it does not skip any pixels, 
and that is because it is 8-bit not 1-bit (in 1-bit images many 
values are extremal and thus can be inferred to have no effect, but in generic 8-bit 
images few pixels can be expected to afford such an optimization, and so it is not
attempted). 

                this      nppi       cpu   description
    case_5        49       722        63   127x127, 1 px set, 47 px dilation
    case_2        36      1934      3614   1024x1024, 3 px set, 10 px dilation
    case_4       274    356468     37950   3333x3333, 5 px set, 47 px dilation
    case_1       628   2635305     59214   4096x4096, 3 px set, 100 px dilation
    case_3       640    355224    135669   3333x3333, 5 large regions set, 47 px dilation

TODO: systematic study of scaling with size of image, number and distribution of pixels initially set, dilation radius, and more.
