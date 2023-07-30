import cv2
import numpy as np
import json
import os.path as path

import multiprocessing as mp

import warnings

from sklearn.cluster import MiniBatchKMeans
from flask import Flask, Response, request

BATCH_SIZE = 20 # How many frames are processed at once
COLOR_ACCURACY = 50 # How many colors are allowed for 1 pixel

app = Flask(__name__)

warnings.filterwarnings("ignore", message="The default value of `n_init` will change from 3 to 'auto' in 1.4.")

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# Define a function to convert an RGB color to hex format
def process_pixel(pixel):
    if np.isscalar(pixel):
        # Handle scalar input values
        return '%02x%02x%02x' % (pixel, pixel, pixel)
    else:
        # Handle RGB color tuples
        r, g, b = pixel
        return '%02x%02x%02x' % (r, g, b)


def quantize_colors(image, num_colors):
    # Convert the image from the RGB color space to the L*a*b* color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Reshape the image into a feature vector
    h, w = image.shape[:2]
    image = image.reshape((h * w, 3))
    
    # Apply k-means using the specified number of clusters
    clt = MiniBatchKMeans(n_clusters=num_colors)
    labels = clt.fit_predict(image)
    
    # Create the quantized image based on the predictions
    quantized = clt.cluster_centers_.astype("uint8")[labels]
    
    # Reshape the feature vectors to images
    quantized = quantized.reshape((h, w, 3))
    
    # Convert from L*a*b* to RGB
    quantized = cv2.cvtColor(quantized, cv2.COLOR_Lab2RGB)
    
    return quantized


def retrieve_pixels(frame, gray_scale, HEIGHT, WIDTH):
    resized_frame = cv2.resize(frame, (WIDTH, HEIGHT))
    
    if gray_scale:
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        pixels = gray_frame.flatten().tolist()
    else:
        quantized_frame = quantize_colors(resized_frame, COLOR_ACCURACY)
        pixels = [process_pixel(pixel) for pixel in quantized_frame.reshape(-1, 3)]
    
    return pixels

# Define a route to get the pixel colors of a video
@app.route('/get_pixels')
def get_pixels():

    gray_scale = str(request.args.get('gray_scale', False)) == 'true'
    video = str(request.args.get('file', 'test.mp4'))

    WIDTH = int(request.args.get('width', 100))
    HEIGHT = int(request.args.get('height', 100))

    video_path_v1 = f"cache/{video}_{WIDTH}x{HEIGHT}_v1.json"
    video_path_v2 = f"cache/{video}_{WIDTH}x{HEIGHT}_v2.json"

    print(gray_scale)
    print(video)

    print(WIDTH)
    print(HEIGHT)

    if (path.exists(video_path_v2)):
        _file = open(video_path_v2, "r")
        frames = json.loads(_file.read())
        frames_data = json.dumps(frames)
        _file.close()
        return Response(frames_data, mimetype='application/json')

    cap = cv2.VideoCapture(video)

    frames = []

    print("starting convertion!")

    if (path.exists(video_path_v1)):
        _file = open(video_path_v1, "r")
        frames = json.loads(_file.read());
        _file.close();
    else:
        with mp.Pool() as pool:
            futures = []

            while True:
                ret, frame = cap.read()
                # print(ret, frame);
                if not ret:
                    break

                future = pool.apply_async(retrieve_pixels, args=(frame, gray_scale, HEIGHT, WIDTH))
                futures.append(future)
                # print(future);

                if len(futures) == BATCH_SIZE:
                    for future in futures:
                        frames.append(future.get())

                    futures = []
    
    print("Finished part 1/2!")

    if (not path.exists(video_path_v1)):
        _file = open(video_path_v1, "w");
        _file.write(json.dumps(frames));
        _file.close();

    newFrames = [];
    lastUpdatedPixel = [];
    frameCount = 0;
    
    for frame in frames:
        i = 0;
        currentFrame = [];
        printProgressBar(frameCount, len(frames), prefix = 'Progress:', suffix = f"Complete ({frameCount}/{len(frames)})", length = 50)
        for pixel in frame:
            if (len(lastUpdatedPixel) < (i+1) or lastUpdatedPixel[i] is None or lastUpdatedPixel[i] != pixel):
                lastUpdatedPixel.insert(i, pixel);
                currentFrame.insert(i, pixel);
            i += 1;
        newFrames.append(currentFrame);
        frameCount += 1;
    
    printProgressBar(len(frames), len(frames), prefix = 'Progress:', suffix = 'Complete', length = 50)
    print("Finished part 2/2!")

    frames_data = json.dumps(newFrames);

    if (not path.exists(video_path_v2)):
        _file = open(video_path_v2, "w");
        _file.write(frames_data);
        _file.close();
    
    cap.release()

    return Response(frames_data, mimetype='application/json')


if __name__ == '__main__':
    app.run()
