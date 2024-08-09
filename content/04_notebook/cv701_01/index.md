---
title: "Gray-algorithms and histogram equalization"
slug: "assignment01"
date: 2023-09-10T14:31:13+02:00
draft: false
readTime: false
comments: false
math: true

links:
  - title: Complete PDF report
    description: Click on the link to download the full report
    website: ./Assignment_1_CV.pdf

---

**!!AT THE VERY END OF THE PAGE YOU CAN FIND THE COMPLETE PDF REPORT!!**

## 1. Histogram equalization

We can achieve histogram equalization in two main ways:

- Equalize every channel in the RGB image and then stack them together again
- Convert the image into a format where the luminance is isolated perform histogram normalization on that channel and then convert back the image to RGB

### RGB channels equalization


```python
# Some code to create the histograms and images 
def create_histogram_rgb(img=None):        
    fig, axs = plt.subplots(1, 4)
    fig.suptitle('RGB Histogram and corresponding image')
    fig.set_size_inches(18.5, 5.5)

    color = ('red','green','blue') 

    cdfs = np.empty(shape=(3,256))
    hists = []
    tot_max_cdf = 0

    for i,col in enumerate(color):
        hist,_ = np.histogram(img[:,:,i].flatten(),256,[0,256])
        cdf = hist.cumsum()
        hists.append(hist)        
        cdfs[i] = cdf               
        if(cdf.max() > tot_max_cdf):
            tot_max_cdf = cdf.max()    

    for i,col in enumerate(color):        
        cdf_normalized = cdfs[i] * float(hists[i].max()) / np.max(cdfs.flatten())                     
        axs[i].plot(cdf_normalized, color = 'orange')
        axs[i].hist(img[:,:,i].ravel(),256,[0,256], color = col)        

    axs[0].legend(('cdf','histogram'), loc = 'upper left')  
    axs[1].legend(('cdf','histogram'), loc = 'upper left')  
    axs[2].legend(('cdf','histogram'), loc = 'upper left')  
    axs[3].imshow(img)

    plt.plot()
    plt.show()
```

The most common method to perform histogram equalization is to process every channel in the RGB image separately and then stack them together to form another RGB image.

Following this approach, we performed the first histogram equalization using the built-in function in OpenCV which takes in input an array of values and outputs the equalized array of values. The input and output are of type **np.uint8**


```python
img_path = "../../../images/used_images/light_dark/"
img_name = "DSC_0015.JPG"

# reading the image
img_rgb = read_img(img_path+img_name)
print("Image Shape: ", img_rgb.shape)
```

    Image Shape:  (4000, 6000, 3)



```python
# Creating the histogram for every channel
create_histogram_rgb(img_rgb)
```


    
![png](task_2.2_files/task_2.2_7_0.png)
    



```python
# function to equalize every channel and then put together again

def equalize_rgb_channels():
    img_rgb_equalized = img_rgb.copy()
    r_equal = np.uint8(img_rgb_equalized[:,:,0])
    g_equal = np.uint8(img_rgb_equalized[:,:,1])
    b_equal = np.uint8(img_rgb_equalized[:,:,2])

    r_equal = cv2.equalizeHist(r_equal)
    g_equal = cv2.equalizeHist(g_equal)
    b_equal = cv2.equalizeHist(b_equal)

    return np.stack((r_equal,g_equal,b_equal),axis=2)        

res_rgb_equal = equalize_rgb_channels()

create_histogram_rgb(res_rgb_equal)
```


    
![png](task_2.2_files/task_2.2_8_0.png)
    


In the first figure we can clearly see that none of the histograms resembles a uniform distribution, on the opposite the 3 histograms have a high spike on the left end of the graph, meaning a majority of dark colors and low-intensity is present, as indeed is expected from the big influence of the background.

In the second figure instead, the histograms were stretched to approximate a uniform distribution. That can be seen from the line of the cumulative distribution that is constantly increasing closely following a line.


### HSI intensity equalization

The second technique foresees that the images need first to be transformed into another color space the HSI. In the HSI color space, the three channels are Hue, Saturation and Intensity. In this color space, the intensity is not scattered among three different channels as it happens to be in RGB.


```python
# Some code to create the histograms and images 
def create_histogram_hsi(img_hsi):    

    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Intensity Histogram in HSI color space and corresponding image')
    fig.set_size_inches(18.5, 5.5)
    
    hist,_ = np.histogram(img_hsi[:,:,2].flatten(),256,[0,255])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()    

    axs[0].hist((img_hsi[:,:,2]).ravel(),256,[0,255])
    axs[0].plot(cdf_normalized, color = 'r')
    axs[0].legend(('cdf','histogram'), loc = 'upper left')        
    axs[1].imshow(img_hsi)
    plt.plot()
    plt.show()
```

First of all we read the image and the convert it to HSI color space using the function defined in the helper function at the top of the notebook. We also normalize it to have values between 0 and 255.


```python
# convert previously used image to HSI
img_hsi = rgb_to_HSI(img_rgb)
img_hsi_norm = float_to_int_HSI(img_hsi)
create_histogram_hsi(img_hsi_norm)
```

    /tmp/ipykernel_1185600/3278179033.py:11: RuntimeWarning: invalid value encountered in cast
      img_hsi_norm[:,:,0] = np.uint8((img_hsi_copy[:,:,0] / (2 * np.pi)) * 255)



    
![png](task_2.2_files/task_2.2_14_1.png)
    


We then normalize the image 


```python
img_hsi_equalized = img_hsi_norm.copy()
img_hsi_equalized[:,:,2] = cv2.equalizeHist(img_hsi_equalized[:,:,2])
create_histogram_hsi(img_hsi_equalized)
```


    
![png](task_2.2_files/task_2.2_16_0.png)
    



```python
img_hsi_equalized_float = int_to_float_HSI(img_hsi_equalized)
new_img_rgb = HSI_to_rgb(img_hsi_equalized_float)
new_img_rgb_contrasted = normalize_rgb(new_img_rgb)
create_histogram_rgb(new_img_rgb_contrasted)
```


    
![png](task_2.2_files/task_2.2_17_0.png)
    



```python
def compare(img1, img2, titles):
    res = np.hstack((img1,img2)) #stacking images side-by-side
    plt.figure(figsize=(20,10))
    plt.title('{} vs {}'.format(titles[0],titles[1]))
    plt.imshow(res,cmap='gray')
```


```python
compare(res_rgb_equal, new_img_rgb_contrasted, ["Equalized RGB", "Equalized HSI"])
```


    
![png](task_2.2_files/task_2.2_19_0.png)
    


### The same result was reproduced with another base image to check that the results were similar and coherrent


```python
img_path = "./"
img_name = "lenna.png"

# reading the image
img_rgb_lenna = read_img(img_path+img_name)
print("Image Shape: ", img_rgb_lenna.shape)
```

    Image Shape:  (330, 330, 3)



```python
# convert previously used image to HSI
img_hsi_lenna = rgb_to_HSI(img_rgb_lenna)
img_hsi_norm_lenna = float_to_int_HSI(img_hsi_lenna)
create_histogram_hsi(img_hsi_norm_lenna)
```


    
![png](task_2.2_files/task_2.2_22_0.png)
    



```python
img_hsi_equalized_lenna = img_hsi_norm_lenna.copy()
img_hsi_equalized_lenna[:,:,2] = cv2.equalizeHist(img_hsi_equalized_lenna[:,:,2])
create_histogram_hsi(img_hsi_equalized_lenna)
```


    
![png](task_2.2_files/task_2.2_23_0.png)
    



```python
img_hsi_equalized_float_lenna = int_to_float_HSI(img_hsi_equalized_lenna)
new_img_rgb_lenna = HSI_to_rgb(img_hsi_equalized_float_lenna)
new_img_rgb_contrasted_lenna = normalize_rgb(new_img_rgb_lenna)
create_histogram_rgb(new_img_rgb_contrasted_lenna)
```


    
![png](task_2.2_files/task_2.2_24_0.png)
    


## 2. Contrast streching

we implemented contrast stretching as proposed in chapter 3.2 in the book Digital Image Processing proposed as reading material in CV701. In this case we use two values namely alpha1 and alpha2, that delimits the starting point and endingn point of the streching zone in the input mapping. We then create a linear function that maps the input to an output in the range defined by beta1 and beta2. The full code is implemented in following cell.

![Alt text](./image.png)


```python
# the main function to stretch the image -> general

def contrast_stretch_channel(image, alpha, beta):

    if len(image.shape) > 2:
        raise Exception("Image must be of only one channel")

    if alpha[0] < 0 or alpha[1] < 0 or beta[0] < 0 or beta[1] < 0:
        raise Exception("Alpha and beta must be positive")
    
    if alpha[0] > 255 or alpha[1] > 255 or beta[0] > 255 or beta[1] > 255:
        raise Exception("Alpha and beta must be smaller than 256")

    max_val = 255 # fixed for 8 bit images    

    c = image.copy()
    cn = np.zeros_like(c)

    alpha1 = alpha[0]
    alpha2 = alpha[1]

    beta1 = beta[0]
    beta2 = beta[1]
    
    img_bit_mask_1 = c <= alpha1    
    img_bit_mask_2 = (alpha1 < c) & (c < alpha2)    
    img_bit_mask_3 = alpha2 <= c    

    s1 = (beta1 / (alpha1 + 1E-20))
    s2 = ((beta2 - beta1) / (alpha2 - alpha1 + 1E-20))
    b2 = beta2 - alpha2 * s2
    s3 = ((max_val - beta2) / (max_val - alpha2 + 1E-20))
    b3 = max_val - max_val * s3
    
    cn[img_bit_mask_1] = c[img_bit_mask_1] * s1            
    cn[img_bit_mask_2] = c[img_bit_mask_2] * s2 + b2        
    cn[img_bit_mask_3] = c[img_bit_mask_3] * s3 + b3    

    return cn

def contrast_stretch_RGB(image, alpha, beta):

    img = image.copy()

    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]

    rn = np.zeros_like(r)
    gn = np.zeros_like(g)
    bn = np.zeros_like(b)    

    rn = contrast_stretch_channel(r, alpha, beta)
    gn = contrast_stretch_channel(g, alpha, beta)
    bn = contrast_stretch_channel(b, alpha, beta)

    return np.stack([rn, gn, bn], axis=2)

```

Reading the image and plot the original hsitograms


```python
img_path = "../../../images/"
img_name = "example3.png"

img_rgb = read_img(img_path + img_name)
create_histogram_rgb(img_rgb)
```


    
![png](task_2.2_files/task_2.2_28_0.png)
    



```python
img_path = "./"
img_name = "lenna.png"

img_rgb = read_img(img_path + img_name)
create_histogram_rgb(img_rgb)
```


    
![png](task_2.2_files/task_2.2_29_0.png)
    


We start our series of exepriments by changing the values of alpha and beta.

1. Leaving alpha and beta with the same values should give us the same exact image, since every intensity is mapped to the same intensity
2. We decrease the range of beta and shift it to the right end of the image. This should produce a very light image, indeed.
3. The next to images we decrease the range of the values in output and we shift them to the left-end of the histogram, so the image becomes even darker


```python
img_stretched = contrast_stretch_RGB(img_rgb, [10, 255], [10, 255])
create_histogram_rgb(img_stretched)
```


    
![png](task_2.2_files/task_2.2_31_0.png)
    



```python
img_stretched = contrast_stretch_RGB(img_rgb, [0, 255], [200, 255])
create_histogram_rgb(img_stretched)
```


    
![png](task_2.2_files/task_2.2_32_0.png)
    



```python
img_stretched = contrast_stretch_RGB(img_rgb, [0, 255], [0, 150])
create_histogram_rgb(img_stretched)
```


    
![png](task_2.2_files/task_2.2_33_0.png)
    



```python
img_stretched = contrast_stretch_RGB(img_rgb, [0, 255], [0, 50])
create_histogram_rgb(img_stretched)
```


    
![png](task_2.2_files/task_2.2_34_0.png)
    



```python
img_stretched = contrast_stretch_RGB(img_rgb, [20, 235], [50, 255])
create_histogram_rgb(img_stretched)
```


    
![png](task_2.2_files/task_2.2_35_0.png)
    



```python
img_stretched = contrast_stretch_RGB(img_rgb, [50, 200], [0, 255])
create_histogram_rgb(img_stretched)
```


    
![png](task_2.2_files/task_2.2_36_0.png)
    


### Clipping
When alpha1 and alpha2 are arbitrary values but beta1 = 0 and beta2 = 255


```python
img_stretched = contrast_stretch_RGB(img_rgb, [50, 150], [0, 255])
create_histogram_rgb(img_stretched)
```


    
![png](task_2.2_files/task_2.2_38_0.png)
    


given the modularity of the function **stretch_channel** we can also modify each channel with its own alpha and value, to get the most contrast out of the image. In this way we have more control over the stretching parameters and it can be calibrated to each min and max of each channel for a better outcome.


```python
def custom_contrast_stretch_RGB(image, alphas, betas):

    img = image.copy()

    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]

    rn = np.zeros_like(r)
    gn = np.zeros_like(g)
    bn = np.zeros_like(b)    

    rn = contrast_stretch_channel(r, alphas[0], betas[0])
    gn = contrast_stretch_channel(g, alphas[1], betas[1])
    bn = contrast_stretch_channel(b, alphas[2], betas[2])

    return np.stack([rn, gn, bn], axis=2)

alphas = [[60,255], [0,230], [45,220]]
betas = [[0,255], [0,255], [0,255]]

custom_img_stretched = custom_contrast_stretch_RGB(img_rgb, alphas, betas)
create_histogram_rgb(custom_img_stretched)
```


    
![png](task_2.2_files/task_2.2_40_0.png)
    


Last the process can also be repeated for an HSI image, by stretching the I channel. Implmenting this as follows:


```python
create_histogram_hsi(img_hsi_norm)
```


    
![png](task_2.2_files/task_2.2_42_0.png)
    



```python
def plot_his_and_rgb(image, alpha, beta):
    stretched_hsi_imgage = image.copy()
    stretched_hsi_imgage[:,:,2] = contrast_stretch_channel(stretched_hsi_imgage[:,:,2], alpha, beta)
    create_histogram_hsi(stretched_hsi_imgage)

    img_hsi_equalized_float = int_to_float_HSI(stretched_hsi_imgage)
    new_img_rgb = HSI_to_rgb(img_hsi_equalized_float)
    new_img_rgb = normalize_rgb(new_img_rgb)
    create_histogram_rgb(new_img_rgb)

    return new_img_rgb
```


```python
new_hsi_stretched = plot_his_and_rgb(img_hsi_norm, [0, 150], [0, 240])
```


    
![png](task_2.2_files/task_2.2_44_0.png)
    



    
![png](task_2.2_files/task_2.2_44_1.png)
    



```python
plot_his_and_rgb(img_hsi_norm, [0, 50], [0, 245])
```


    
![png](task_2.2_files/task_2.2_45_0.png)
    



    
![png](task_2.2_files/task_2.2_45_1.png)
    


## Thresholding 

Thresholding is another particular case of contrast stretching. It is when both alpha are the same and beta are the extremes, namely 0-255. 
Some experiments were performed with different thresholds


```python
bw_image = img_rgb.copy()
bw_image = cv2.cvtColor(bw_image, cv2.COLOR_RGB2GRAY)

alphas = [[50,50], [100, 100], [150, 150]]

# stack images side-by-side

fig, axs = plt.subplots(1, 3)
fig.suptitle('Thresholding with different alpha values')
fig.set_size_inches(18.5, 5.5)

for i, alpha in enumerate(alphas):    
    #axs[0].legend((alpha), loc = 'upper left')        
    axs[i].imshow(contrast_stretch_channel(bw_image, alpha, [0, 255]), cmap='gray')

plt.plot()

```




    []




    
![png](task_2.2_files/task_2.2_47_1.png)
    



```python
### Comparing the results of the different methods


compare(custom_img_stretched, new_img_rgb_contrasted_lenna, ["Image maximally stretched on different channels RGB", "Histogram Equalized image on HSI"])


```


    
![png](task_2.2_files/task_2.2_48_0.png)
    



```python
### Comparing the results of the different methods
compare(new_img_rgb_contrasted, new_hsi_stretched, ["Histogram Equalized HSI image", "Contrast stretched image on HSI"])

```


    
![png](task_2.2_files/task_2.2_49_0.png)
    



```python
# some general and helper functions

def float_to_int_HSI(img_hsi):
    # Expected input:
    #   channel 1 => H => [0, 2pi]
    #   channel 2 => S => [0, 1]
    #   channel 3 => I => [0, 1]
    
    img_hsi_copy = img_hsi.copy()
    img_hsi_norm = np.zeros(shape=(img_hsi_copy.shape), dtype=np.uint8)
    img_hsi_norm[:,:,0] = np.uint8((img_hsi_copy[:,:,0] / (2 * np.pi)) * 255)
    img_hsi_norm[:,:,1] = np.uint8(img_hsi_copy[:,:,1] * 255)
    img_hsi_norm[:,:,2] = np.uint8(img_hsi_copy[:,:,2] * 255)

    return img_hsi_norm

def int_to_float_HSI(img_hsi):
    # Expected input:
    #   channel 1 => H => [0, 255]
    #   channel 2 => S => [0, 255]
    #   channel 3 => I => [0, 255]
    
    img_hsi_copy = img_hsi.copy()
    img_hsi_norm = np.zeros(shape=(img_hsi_copy.shape), dtype=np.float32)
    img_hsi_norm[:,:,0] = (img_hsi_copy[:,:,0] / 255) * (2 * np.pi)
    img_hsi_norm[:,:,1] = img_hsi_copy[:,:,1] / 255
    img_hsi_norm[:,:,2] = img_hsi_copy[:,:,2] / 255

    return img_hsi_norm

def normalize_rgb(img_rgb):
    img_rgb_copy = img_rgb.copy()
    return np.uint8(img_rgb_copy * 255)

def read_img(img_path):
    return imageio.v3.imread(img_path)

def HSI_to_rgb(img):
    h = img[:,:, 0] # expected range [0, 2pi] we actually should get from 0 to 1 as well
    s = img[:,:, 1] # expected range [0, 1]
    i = img[:,:, 2] # expected range [0, 1]

    h1 = h < 1/3 * 2 * np.pi
    h2 = (1/3 * 2 * np.pi <= h) & (h < 2/3 * 2 * np.pi)
    h3 = 2/3 * 2 * np.pi <= h     

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)
    h_curr = np.zeros_like(h)

    # case 1
    h_curr[h1] = h[h1]
    b[h1] = i[h1] * (1 - s[h1])
    r[h1] = i[h1] * (1 + (s[h1] * np.cos(h_curr[h1])) / (np.cos(np.pi/3 - h_curr[h1])))
    g[h1] = 3 * i[h1] - (r[h1] + b[h1])

    # case 2    
    h_curr[h2] = h[h2] - (2/3 * np.pi)
    r[h2] = i[h2] * (1 - s[h2])
    g[h2] = i[h2] * (1 + (s[h2] * np.cos(h_curr[h2])) / (np.cos(np.pi/3 - h_curr[h2])))
    b[h2] = 3 * i[h2] - (r[h2] + g[h2])

    # case 3
    h_curr[h3] = h[h3] - (4/3 * np.pi)
    g[h3] = i[h3] * (1 - s[h3])
    b[h3] = i[h3] * (1 + (s[h3] * np.cos(h_curr[h3])) / (np.cos(np.pi/3 - h_curr[h3])))
    r[h3] = 3 * i[h3] - (g[h3] + b[h3])    

    r = r / max(r.flatten())
    g = g / max(g.flatten())
    b = b / max(b.flatten())
    
    return np.stack((r,g,b),axis=2)

def rgb_to_HSI(img):

    with np.errstate(divide="ignore", invalid="ignore"):

        # load image with 32 floating point type
        rgb = np.float32(img)/255.0

        # Separating RGB    
        r = rgb[:,:,0]
        g = rgb[:,:,1]
        b = rgb[:,:,2]        

        # Intensity
        intensity = (r+b+g)/3

        # Saturation
        min = np.minimum(np.minimum(r,g),b)
        sat = 1 - ((3/(r+g+b+0.0000001))*min)

        # Hue
        hue = np.arccos((0.5*((r-g)+(r-b)))/((np.sqrt((r-g)**2+(r-b)*(g-b)))))
        hue[b>g] = 2*np.pi - hue[b>g]        

        return np.stack((hue,sat,intensity),axis=2)     
```