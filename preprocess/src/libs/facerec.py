import dlib
import numpy as np
import face_recognition_models
import dlib.cuda as cuda


class FaceRec():

    def __init__(self, gpu):
        cuda.set_device(gpu)
        face_detector = dlib.get_frontal_face_detector()

        predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
        pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

        cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

        face_recognition_model = face_recognition_models.face_recognition_model_location()
        face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

    def _rect_to_css(self, rect):
        return rect.top(), rect.right(), rect.bottom(), rect.left()

    def _css_to_rect(css):
        return dlib.rectangle(css[3], css[0], css[1], css[2])

    def _trim_css_to_bounds(self, css, image_shape):
        return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

    def face_locations(img, number_of_times_to_upsample=0):
        return [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) for face in cnn_face_detector(img, number_of_times_to_upsample, batch_size=32)]

    def face_landmarks(face_image, face_locations=None):
        if face_locations is None:
            face_locations = face_locations(face_image)
        else:
            face_locations = [_css_to_rect(face_location) for face_location in face_locations]

        pose_predictor = pose_predictor_68_point
        return [pose_predictor(face_image, face_location) for face_location in face_locations]

    def face_encodings(face_image, known_face_locations=None, num_jitters=1):
        raw_landmarks = face_landmarks(face_image, known_face_locations)
        return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]

    def _raw_face_locations_batched(self, images, number_of_times_to_upsample=1, batch_size=32):
        x = self.cnn_face_detector(images, number_of_times_to_upsample, batch_size=batch_size)
        return x

    def batch_face_locations(self, images, number_of_times_to_upsample=1, batch_size=32):
        def convert_cnn_detections_to_css(detections):
            return [self._trim_css_to_bounds(self._rect_to_css(face.rect), images[0].shape) for face in detections]

        raw_detections_batched = self._raw_face_locations_batched(images, number_of_times_to_upsample, batch_size)
        return list(map(convert_cnn_detections_to_css, raw_detections_batched))
