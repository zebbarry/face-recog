import face_recognition as fr
import os
import cv2
import argparse
import numpy as np
from time import sleep, time
from imutils.video import FPS
import colours

# Colours for bounding box and name on image
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)

# Scales for processing and displaying image
SCALE = 60
DISPLAY_SCALE = 60

# Tolerance for matching faces
TOLERANCE = 0.6


def get_encoded_faces(path):
    """
    Looks through the folder and encodes all
    the faces alongside their name

    :path: path to folder of previously identified faces
    :return: dict of (name, image encoded)
    """
    encoded = {}

    dirpath, dnames, fnames = next(os.walk(path))
    for f in fnames:
        if f.endswith(".jpg") or f.endswith(".png"):
            # Load image and get embeddings
            face = fr.load_image_file(os.path.join(dirpath, f))
            encoding = fr.face_encodings(face)
            if len(encoding) != 0:
                encoded[f.split(".")[0]] = encoding[0]
            else:
                # No faces found in image
                colours.setText(colours.RED)
                print("[ERROR] No faces in " + f)
                colours.setText(colours.GREEN)

    return encoded


def classify_face(faces_encoded, known_face_names, face_encoding):
    """
    Classifies a given encoding if a match is found. Returns none is no match found

    :faces_encoded: list of known face encodings
    :known_face_names: list of face names to match known encodings
    :face_encoding: unknown face encoding to be identified
    :return: tuple of (name, closest distance)
    """
    name = None
    if faces_encoded:
        # Use the known face with the smallest distance to the new face
        face_distances = fr.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if face_distances[best_match_index] <= TOLERANCE:
            name = known_face_names[best_match_index]

    return name, face_distances[best_match_index]


def test(test_set, base_version):
    """
    Tests classification method on Yale Faces dataset using folders filled with images of that subject.

    :test_set: file path to Yale Faces dataset
    :base_version: version of subject image to use as reference
    :return: Accuracy over entire dataset
    """
    dataset = {}

    # Read all images into dicts based on subject
    colours.setText(colours.GREEN)
    print("[INFO] Processing images...")
    dirpath, dnames, fnames = next(os.walk(test_set))
    for d in sorted(dnames):
        dataset[d] = get_encoded_faces(os.path.join(dirpath, d))

    # Seperate base image from the rest
    print("[INFO] Identified base image as: ", base_version)
    known_face_names = []
    faces_encoded = []
    for subject, encodings in sorted(dataset.items()):
        known_face_names.append(subject)
        for im_version, encoding in encodings.items():
            if base_version in im_version:
                faces_encoded.append(encoding)

    # Iterate through test set and check if image is recognized
    print("[INFO] Recognizing test images...")
    summary = {}
    for subject, encodings in dataset.items():
        total = len(encodings)
        results = {}
        correct = 0
        for im_version, encoding in encodings.items():
            name, certainty = classify_face(faces_encoded, known_face_names, encoding)
            results[im_version.split("_")[1]] = 1 if subject == name else name
            correct += (subject == name)
        summary[subject] = ((correct * 100) / total, results)

    # Displaying results
    colours.setText(colours.BLUE)
    sum_percent = 0
    total_images = 0
    for subject, (percent, results) in sorted(summary.items()):
        print("[RESULTS] {}: {:.2f}% ({}/{})".format(subject, percent, percent / 100 * len(results), len(results)))
        sum_percent += percent
        total_images += len(results)
        for im_version, result in sorted(results.items()):
            if result != 1:
                colours.setText(colours.RED)
                print("[ERROR] \t{} identified as: {}".format(im_version, result))
                colours.setText(colours.BLUE)

    unique_faces = len(summary)
    print("[RESULTS] Total accuracy over {} images of {} unique faces: {:.2f}%".format(total_images, unique_faces,
                                                                                       sum_percent / unique_faces))
    return sum_percent / unique_faces


def label_face(img, bounding_box, name, scalar=1):
    """
    Draws bounding box around face and labels with name.

    :img: Image
    :bounding_box: tuple of box corners
    :name: string of face name
    :scalar: ratio of proccesing size to display size of image
    :return: Image with rectange and name drawn
    """
    # Draw a box around the face
    (top, right, bottom, left) = bounding_box
    top = int(top * scalar)
    right = int(right * scalar)
    left = int(left * scalar)
    bottom = int(bottom * scalar)
    cv2.rectangle(img, (left, top), (right, bottom), BLUE, 2)

    # Draw a label with a name below the face
    cv2.rectangle(img, (left - 1, top), (right + 1, top - 20), BLUE, cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, name, (left, top - 5), font, 0.6, GREEN, 2)

    return img


def label_image_from_file(file, faces_dir, scale=SCALE):
    """
    Classifies and labels all faces in an image.

    :scale: int percentage of original size to process image
    :file: string of image file path
    :faces_dir: directory with prelabelled images of faces
    :return: list of face names in image
    """
    # Get encodings for previously identified faces
    colours.setText(colours.GREEN)
    print("[INFO] Processing existing images...")
    faces_encoded = []
    known_face_names = []
    faces = get_encoded_faces(faces_dir)
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())
    print("[INFO] Recognized faces ({}): ".format(len(known_face_names)), ", ".join(known_face_names))

    # Read image to be labelled
    print("[INFO] Reading test image...")
    print("[INFO] Detection scale: ", scale)
    img = cv2.imread(file, 1)
    small_dim = (int(img.shape[1] * scale / 100), int(img.shape[0] * scale / 100))
    img = cv2.resize(img, small_dim, interpolation=cv2.INTER_AREA)

    # Find faces in image
    print("[INFO] Processing test image...")
    face_locations = fr.face_locations(img)
    unknown_face_encodings = fr.face_encodings(img, face_locations)

    if len(face_locations) > 0:
        # Compare each face and identify
        unknown_index = 1
        face_names = []
        for face_encoding in unknown_face_encodings:
            name, certainty = classify_face(faces_encoded, known_face_names, face_encoding)
            if name is not None:
                face_names.append((name, certainty))
            else:
                # Unknown face found
                name = "Face #" + str(unknown_index)
                face_names.append((name, certainty))
                unknown_index += 1
                faces_encoded.append(face_encoding)
                known_face_names.append(name)

        # Label faces on image
        for bounding_box, (name, certainty) in zip(face_locations, face_names):
            img = label_face(img, bounding_box, name)
    else:
        face_names = [("None", 0)]

    colours.setText(colours.BLUE)
    print("[RESULTS] Current faces: ", ", ".join([face_names[i][0] for i in range(len(face_names))]))
    print("[RESULTS] Total faces encoded ({}): ".format(len(known_face_names)), ", ".join(known_face_names))

    # Display the resulting image
    cv2.imshow(file, img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    return face_names


def label_video(faces_dir, scale=SCALE, display_scale=DISPLAY_SCALE, video=0):
    """
    Finds all the faces in an video feed and labels them. Prints result fps info and current faces.

    
    :faces_dir: directory with prelabelled images of faces
    :scale: percentage of original that image should be processed at.
    :display_scale: percentage of original that image should be displayed as.
    """
    colours.setText(colours.GREEN)
    print("[INFO] Processing existing images...")
    faces_encoded = []
    known_face_names = []
    faces = get_encoded_faces(faces_dir)
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())
    print("[INFO] Recognized faces ({}): ".format(len(known_face_names)), ", ".join(known_face_names))
    print("[INFO] Detection scale: ", scale)
    print("[INFO] Viewing scale: ", display_scale)

    # initialize the video stream, then allow the camera sensor to warm up
    print("[INFO] Starting video stream")
    cap = cv2.VideoCapture(video)
    sleep(2.0)

    ret, img = cap.read()
    small_dim = (int(img.shape[1] * scale / 100), int(img.shape[0] * scale / 100))
    full_dim = (int(img.shape[1] * display_scale / 100), int(img.shape[0] * display_scale / 100))

    unknown_index = 1
    process_frame = True

    # start the FPS throughput estimator
    print("[INFO] Processing output from video stream...")
    fps = FPS().start()

    while cap.isOpened():

        ret, full_img = cap.read()

        if not ret:
            print("[INFO] No image returned")
            break

        small_img = cv2.resize(full_img, small_dim, interpolation=cv2.INTER_AREA)
        full_img = cv2.resize(full_img, full_dim, interpolation=cv2.INTER_AREA)

        if process_frame:
            # Record frames
            fps.update()

            # Find Faces
            face_locations = fr.face_locations(small_img)
            unknown_face_encodings = fr.face_encodings(small_img, face_locations)

            # Compare each face and identify
            face_names = []
            for face_encoding in unknown_face_encodings:
                name, certianty = classify_face(faces_encoded, known_face_names, face_encoding)
                if name != None:
                    face_names.append(name)
                else:
                    name = "Face #" + str(unknown_index)
                    unknown_index += 1
                    face_names.append(name)
                    faces_encoded.append(face_encoding)
                    known_face_names.append(name)

        # Label faces on image
        for bounding_box, name in zip(face_locations, face_names):
            full_img = label_face(full_img, bounding_box, name, display_scale / scale)

        # Display the resulting image
        cv2.imshow('Video', full_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Process every second frame for faster fps
        # process_frame = not process_frame

    # stop the timer and display FPS information
    fps.stop()
    cap.release()
    cv2.destroyAllWindows()
    colours.setText(colours.RED)
    print("[RESULTS] elasped time: {:.2f}".format(fps.elapsed()))
    print("[RESULTS] approx. FPS: {:.2f}".format(fps.fps()))
    colours.setText(colours.BLUE)
    print("[RESULTS] Current faces: ", ", ".join(face_names))
    print("[RESULTS] Total faces encoded ({}): ".format(len(known_face_names)), ", ".join(known_face_names))


def main():
    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default=None,
                    help="Path to image for classification")
    ap.add_argument("-s", "--scale", type=int, default=SCALE,
                    help="Downsize scale processing percentage")
    ap.add_argument("-d", "--display_scale", type=int, default=DISPLAY_SCALE,
                    help="Downsize scale display percentage")
    ap.add_argument("-f", "--faces_dir", default="faces",
                    help="Pre-recorded face directory")
    ap.add_argument("-t", "--test_dir", default=None,
                    help="Directory of yalefaces dataset for testing")
    ap.add_argument("-r", "--reference", default="_normal",
                    help="Type of image in yalefaces dataset to be used as reference")
    ap.add_argument("-v", "--video", default=0,
                    help="Input video")
    args = vars(ap.parse_args())

    if args["image"] is not None:
        # Label image given
        label_image_from_file(args["image"], args["faces_dir"], args["scale"])
    elif args["test_dir"] is not None:
        # Test method on Yale Faces dataset
        test(args["test_dir"], args["reference"])
    else:
        # Read reltime video from camera and label
        label_video(args["faces_dir"], args["scale"], args["display_scale"], args["video"])


if __name__ == "__main__":
    main()
    colours.setText(colours.RESET)
