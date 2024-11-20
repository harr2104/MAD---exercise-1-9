// import React, { useState } from 'react';
// import { View, Button, Text, TextInput, StyleSheet } from 'react-native';
// import * as Speech from 'expo-speech';
// import { Audio } from 'expo-av';

// export default function Exp6() {
//   const [textToSpeak, setTextToSpeak] = useState('');
//   const [speechText, setSpeechText] = useState('');
//   const [recording, setRecording] = useState(null);
  
//   // Text-to-Speech Functionality
//   const speakText = () => {
//     Speech.speak(textToSpeak);
//   };

//   // Start Recording (Speech-to-Text Alternative)
//   const startRecording = async () => {
//     try {
//       console.log('Requesting permissions..');
//       await Audio.requestPermissionsAsync();
//       await Audio.setAudioModeAsync({
//         allowsRecordingIOS: true,
//         playsInSilentModeIOS: true,
//       });

//       console.log('Starting recording..');
//       const recording = new Audio.Recording();
//       await recording.prepareToRecordAsync(Audio.RECORDING_OPTIONS_PRESET_HIGH_QUALITY);
//       await recording.startAsync();
//       setRecording(recording);
//       console.log('Recording started');
//     } catch (err) {
//       console.error('Failed to start recording', err);
//     }
//   };

//   // Stop Recording
//   const stopRecording = async () => {
//     console.log('Stopping recording..');
//     setRecording(undefined);
//     await recording.stopAndUnloadAsync();
//     const uri = recording.getURI();
//     console.log('Recording stopped and stored at', uri);

//     // Placeholder: Upload the audio file to a speech-to-text API
//     // Here you would send `uri` to a speech recognition API like Google Speech-to-Text
//     setSpeechText('Audio recording complete. Implement API call to transcribe speech.');
//   };

//   return (
//     <View style={styles.container}>
//       {/* Text-to-Speech Section */}
//       <Text style={styles.heading}>Text-to-Speech (TTS)</Text>
//       <TextInput
//         style={styles.input}
//         placeholder="Enter text to speak"
//         value={textToSpeak}
//         onChangeText={setTextToSpeak}
//       />
//       <Button title="Speak Text" onPress={speakText} />

//       {/* Speech-to-Text Section */}
//       <Text style={[styles.heading, { marginTop: 20 }]}>Speech-to-Text (STT)</Text>
//       <Button
//         title={recording ? 'Stop Recording' : 'Start Recording'}
//         onPress={recording ? stopRecording : startRecording}
//       />
//       <Text style={styles.resultText}>{speechText}</Text>
//     </View>
//   );
// }

// const styles = StyleSheet.create({
//   container: {
//     flex: 1,
//     justifyContent: 'center',
//     alignItems: 'center',
//     padding: 20,
//   },
//   input: {
//     borderWidth: 1,
//     padding: 10,
//     marginVertical: 20,
//     width: '100%',
//   },
//   heading: {
//     fontSize: 18,
//     fontWeight: 'bold',
//     marginBottom: 10,
//   },
//   resultText: {
//     marginTop: 20,
//     fontSize: 16,
//   },
// });

import React, { useState } from 'react';
import { View, Button, Text, TextInput, StyleSheet } from 'react-native';
import * as Speech from 'expo-speech';
import { Audio } from 'expo-av';

export default function Exp6() {
  const [textToSpeak, setTextToSpeak] = useState('');
  const [speechText, setSpeechText] = useState('');
  const [recording, setRecording] = useState(null);

  // Replace this with your Google API Key
  const GOOGLE_CLOUD_API_KEY = 'AIzaSyC3FmC56DsAd-UPkD7SWQCmZPHl07Mi_fo';

  // Text-to-Speech Functionality
  const speakText = () => {
    if (textToSpeak.trim() !== '') {
      Speech.speak(textToSpeak);
    } else {
      alert('Please enter text to speak');
    }
  };

  // Start Recording (Speech-to-Text Alternative)
  const startRecording = async () => {
    try {
      console.log('Requesting permissions...');
      await Audio.requestPermissionsAsync();
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      console.log('Starting recording...');
        const recording = new Audio.Recording();
      await recording.prepareToRecordAsync(Audio.RECORDING_OPTIONS_PRESET_HIGH_QUALITY);
      await recording.startAsync();

      console.log('Recording started');
    } catch (err) {
      console.error('Failed to start recording', err);
    }
  };

  // Stop Recording and Send to Google Speech-to-Text API
  const stopRecording = async () => {
    try {
      console.log('Stopping recording...');
      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();
      console.log('Recording stopped and stored at', uri);
        console.log(uri)
      setRecording(null); // Set recording to null to indicate it has stopped

      // Upload the audio file to Google Cloud for transcription
      const transcription = await transcribeAudio(uri);
      setSpeechText(transcription);
    } catch (error) {
      console.error('Failed to stop recording', error);
    }
  };

  // Function to send audio file to Google Cloud Speech-to-Text API
  const transcribeAudio = async (uri) => {
    try {
      console.log('Fetching audio file...');
      const response = await fetch(uri);
      const blob = await response.blob();
      const base64Audio = await blobToBase64(blob);

      // Payload for Google Cloud Speech-to-Text
      const requestPayload = {
        config: {
          encoding: 'LINEAR16',
          sampleRateHertz: 16000,
          languageCode: 'en-US',
        },
        audio: {
          content: base64Audio,
        },
      };

      // Make request to Google API
      console.log('Sending request to Google Cloud...');
      const responseGoogle = await fetch(
        `https://speech.googleapis.com/v1/speech:recognize?key=${GOOGLE_CLOUD_API_KEY}`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestPayload),
        }
      );

      const data = await responseGoogle.json();

      if (responseGoogle.ok) {
        const transcription = data.results
          .map((result) => result.alternatives[0].transcript)
          .join('\n');

        console.log('Transcription successful:', transcription);
        return transcription;
      } else {
        console.error('Transcription failed', data);
        return 'Error in transcription';
      }
    } catch (error) {
      console.error('Error transcribing audio', error);
      return 'Error in transcription';
    }
  };

  // Helper function to convert Blob to Base64
  const blobToBase64 = (blob) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        resolve(reader.result.split(',')[1]); // Return only the Base64 string part
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  };

  return (
    <View style={styles.container}>
      {/* Text-to-Speech Section */}
      <Text style={styles.heading}>Text-to-Speech (TTS)</Text>
      <TextInput
        style={styles.input}
        placeholder="Enter text to speak"
        value={textToSpeak}
        onChangeText={setTextToSpeak}
      />
      <Button title="Speak Text" onPress={speakText} />

      {/* Speech-to-Text Section */}
      <Text style={[styles.heading, { marginTop: 20 }]}>Speech-to-Text (STT)</Text>
      <Button
        title={recording ? 'Stop Recording' : 'Start Recording'}
        onPress={recording ? stopRecording : startRecording}
      />
      <Text style={styles.resultText}>{speechText}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  input: {
    borderWidth: 1,
    padding: 10,
    marginVertical: 20,
    width: '100%',
  },
  heading: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  resultText: {
    marginTop: 20,
    fontSize: 16,
  },
});

