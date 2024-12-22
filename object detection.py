import cv2
import numpy as np

def segment_image_kmeans(image, k=4):
    # Reshape the image to a 2D array of pixels
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers to uint8 and reshape labels to original image shape
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape(image.shape)
    rgb_segment = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('rgb_segment.jpg', rgb_segment)
    return segmented_image

def is_color_in_range(pixel, lower_bound, upper_bound):
    return np.all(pixel >= lower_bound) and np.all(pixel <= upper_bound)

def is_polish_or_indonesian_flag(image):
    # Segment the image
    segmented_image = segment_image_kmeans(image, k=4)

    # Define color ranges for red and white
    red_lower = np.array([0, 0, 100])
    red_upper = np.array([100, 100, 255])
    white_lower = np.array([200, 200, 200])
    white_upper = np.array([255, 255, 255])

    # Get the height and width of the image
    height, width, _ = segmented_image.shape

    for row in range(height):
        row_colors = segmented_image[row]

        # Count red and white pixels in the row
        red_pixels = np.sum([is_color_in_range(pixel, red_lower, red_upper) for pixel in row_colors])
        white_pixels = np.sum([is_color_in_range(pixel, white_lower, white_upper) for pixel in row_colors])

        # Determine if the row is primarily red or white
        if red_pixels > width // 4:
            # Found a red row, search for a white row below
            for next_row in range(row + 1, height):
                next_row_colors = segmented_image[next_row]
                white_pixels_next = np.sum([is_color_in_range(pixel, white_lower, white_upper) for pixel in next_row_colors])
                if white_pixels_next > width // 4:
                    return "Polish Flag"
            return "No Flag Detected"

        if white_pixels > width // 4:
            # Found a white row, search for a red row below
            for next_row in range(row + 1, height):
                next_row_colors = segmented_image[next_row]
                red_pixels_next = np.sum([is_color_in_range(pixel, red_lower, red_upper) for pixel in next_row_colors])
                if red_pixels_next > width // 4:
                    return "Indonesian Flag"
            return "No Flag Detected"

    return "No Flag Detected"

# Load the image
image_path = "flag.png"  # Replace with your image path
image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to load image. Please check the path.")
else:
    # Resize the image for consistent processing (optional)
    image = cv2.resize(image, (300, 200))

    # Detect the flag
    result = is_polish_or_indonesian_flag(image)
    print(result)

    # Show the segmented image for debugging
    segmented_image = segment_image_kmeans(image, k=4)
    cv2.imshow("Segmented Image", segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

