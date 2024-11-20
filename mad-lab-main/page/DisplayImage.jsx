import React, { useState, useRef, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image, ActivityIndicator } from 'react-native';
import { Camera, CameraView } from 'expo-camera';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import * as FileSystem from 'expo-file-system';
import * as blazeface from '@tensorflow-models/blazeface';
import { decodeJpeg } from '@tensorflow/tfjs-react-native';
import { GLView } from 'expo';
import Expo2DContext from 'expo-2d-context';

export default function RealFaceDetection() {
    const [hasPermission, setHasPermission] = useState(null);
    const [image, setImage] = useState(null);
    const [detections, setDetections] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const cameraRef = useRef(null);
    const [isTfReady, setIsTfReady] = useState(false);
    const [model, setModel] = useState(null);
    const glViewRef = useRef(null);

    useEffect(() => {
        let isMounted = true;
        (async () => {
            const { status } = await Camera.requestCameraPermissionsAsync();
            if (isMounted) setHasPermission(status === 'granted');

            // Load TensorFlow
            await tf.ready();
            if (isMounted) setIsTfReady(true);

            // Load BlazeFace model
            const loadedModel = await blazeface.load();
            if (isMounted) setModel(loadedModel);
        })();

        return () => {
            // Cleanup function
            isMounted = false;
        };
    }, []);

    const takePicture = async () => {
        if (cameraRef.current) {
            const photo = await cameraRef.current.takePictureAsync();
            setImage(photo.uri);
        }
    };

    const detectFaces = async () => {
        if (image && model) {
            setIsLoading(true);
            try {
                const imgB64 = await FileSystem.readAsStringAsync(image, {
                    encoding: FileSystem.EncodingType.Base64,
                });
                const imgBuffer = tf.util.encodeString(imgB64, 'base64').buffer;
                const rawImageData = new Uint8Array(imgBuffer);
                const imageTensor = decodeJpeg(rawImageData);

                // Use the BlazeFace model to detect faces
                const predictions = await model.estimateFaces(imageTensor, false);

                if (predictions.length > 0) {
                    const detectedFaces = predictions.map(prediction => {
                        return {
                            class: 'Hursun',
                            score: prediction.probability[0],
                            boundingBox: prediction.topLeft.concat(prediction.bottomRight),
                            landmarks: [
                                { part: 'left_eye', x: prediction.landmarks[0][0], y: prediction.landmarks[0][1] },
                                { part: 'right_eye', x: prediction.landmarks[1][0], y: prediction.landmarks[1][1] },
                                { part: 'nose', x: prediction.landmarks[2][0], y: prediction.landmarks[2][1] },
                                { part: 'mouth_left', x: prediction.landmarks[3][0], y: prediction.landmarks[3][1] },
                                { part: 'mouth_right', x: prediction.landmarks[4][0], y: prediction.landmarks[4][1] },
                            ],
                        };
                    });
                    setDetections(detectedFaces);
                } else {
                    setDetections([]);
                }
            } catch (error) {
                console.error('Detection Error:', error);
            } finally {
                setIsLoading(false);
            }
        }
    };

    const onContextCreate = (gl) => {
        const ctx = new Expo2DContext(gl);
        if (detections.length > 0) {
            ctx.clearRect(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
            detections.forEach(detection => {
                detection.landmarks.forEach(landmark => {
                    ctx.beginPath();
                    ctx.arc(landmark.x, landmark.y, 5, 0, 2 * Math.PI);
                    ctx.fillStyle = 'red';
                    ctx.fill();
                    ctx.closePath();
                });
            });
            ctx.flush();
        }
    };

    if (hasPermission === null) {
        return <View />;
    }
    if (hasPermission === false) {
        return <Text>No access to camera</Text>;
    }

    return (
        <View style={styles.container}>
            {!image && (
                <CameraView style={styles.camera} ref={cameraRef}>
                    <View style={styles.buttonContainer}>
                        <TouchableOpacity style={styles.button} onPress={takePicture}>
                            <Text style={styles.text}>Take Photo</Text>
                        </TouchableOpacity>
                    </View>
                </CameraView>
            )}
            {image && (
                <View style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Image source={{ uri: image }} style={styles.previewImage} />
                    {/* <GLView
                        style={styles.canvas}
                        onContextCreate={onContextCreate}
                        ref={glViewRef}
                    /> */}
                    <View style={styles.actionButtons}>
                        <TouchableOpacity style={styles.actionButton} onPress={detectFaces}>
                            <Text style={styles.text}>Detect Faces</Text>
                        </TouchableOpacity>
                        <TouchableOpacity style={styles.actionButton} onPress={() => setImage(null)}>
                            <Text style={styles.text}>Capture Another Photo</Text>
                        </TouchableOpacity>
                    </View>
                    {isLoading && <ActivityIndicator size="large" color="#0000ff" />}
                    {detections.length > 0 ? (
                        <View style={styles.detectionResults}>
                            <Text style={styles.detectionText}>Detected Face:</Text>
                            {detections.map((detection, index) => (
                                <View key={index} style={styles.detectionItemContainer}>
                                    <Text style={styles.detectionItem}>
                                        {detection.class} - {Math.round(detection.score * 100)}%
                                    </Text>
                                    <Text style={styles.landmarksText}>Landmarks:</Text>
                                    {detection.landmarks.map((landmark, i) => (
                                        <Text key={i} style={styles.landmarkItem}>
                                            {landmark.part}: ({landmark.x}, {landmark.y})
                                        </Text>
                                    ))}
                                </View>
                            ))}
                        </View>
                    ) : (
                        !isLoading && (
                            <Text style={styles.noObjectText}>
                                No Face detected. Please try another image.
                            </Text>
                        )
                    )}
                </View>
            )}
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#fff',
    },
    camera: {
        flex: 1,
        width: '100%',
    },
    buttonContainer: {
        flex: 1,
        backgroundColor: 'transparent',
        flexDirection: 'row',
        margin: 20,
    },
    button: {
        flex: 0.3,
        alignSelf: 'flex-end',
        alignItems: 'center',
        backgroundColor: '#f08',
        padding: 10,
        borderRadius: 5,
    },
    text: {
        fontSize: 18,
        color: 'white',
    },
    previewImage: {
        width: 300,
        height: 300,
        borderRadius: 10,
        marginBottom: 20,
    },
    canvas: {
        position: 'absolute',
        top: 0,
        left: 0,
        width: 300,
        height: 300,
    },
    actionButtons: {
        flexDirection: 'row',
        justifyContent: 'space-around',
    },
    actionButton: {
        backgroundColor: '#2196F3',
        padding: 10,
        margin: 10,
        borderRadius: 5,
    },
    detectionResults: {
        marginTop: 20,
        padding: 10,
        backgroundColor: '#f0f0f0',
        borderRadius: 10,
    },
    detectionText: {
        fontSize: 18,
        fontWeight: 'bold',
    },
    detectionItemContainer: {
        marginBottom: 10,
    },
    detectionItem: {
        fontSize: 16,
        marginVertical: 2,
    },
    landmarksText: {
        fontSize: 16,
        fontWeight: 'bold',
        marginTop: 5,
    },
    landmarkItem: {
        fontSize: 14,
        marginLeft: 10,
    },
    noObjectText: {
        marginTop: 20,
        fontSize: 16,
        color: 'red',
    },
});
