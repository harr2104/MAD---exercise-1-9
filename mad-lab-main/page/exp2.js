import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Picker } from '@react-native-picker/picker';
import Toast from 'react-native-toast-message';

export default function Exp2() {
  const [fontSize, setFontSize] = useState(16);
  const [textColor, setTextColor] = useState('#000');

  const showToast = () => {
    Toast.show({
      type: 'success',
      position: 'bottom',
      text1: 'Button Pressed!',
      text2: 'The text style has been updated.',
      visibilityTime: 3000,
      autoHide: true,
    });
  };

  return (
    <View style={styles.container}>
      <Text style={[styles.text, { fontSize: fontSize, color: textColor }]}>
        Customize my style!
      </Text>

      <View style={styles.pickerContainer}>
        <Text style={styles.label}>Select Font Size:</Text>
        <Picker
          selectedValue={fontSize}
        //   style={styles.picker}
          onValueChange={(itemValue) => setFontSize(itemValue)}
        >
          <Picker.Item label="Small" value={14} />
          <Picker.Item label="Medium" value={18} />
          <Picker.Item label="Large" value={22} />
        </Picker>
      </View>

      <View style={styles.pickerContainer}>
        <Text style={styles.label}>Select Text Color:</Text>
        <Picker
          selectedValue={textColor}
        //   style={styles.picker}
          onValueChange={(itemValue) => setTextColor(itemValue)}
        >
          <Picker.Item label="Black" value="#000" />
          <Picker.Item label="Red" value="#f00" />
          <Picker.Item label="Green" value="#0f0" />
          <Picker.Item label="Blue" value="#00f" />
        </Picker>
      </View>

      <TouchableOpacity style={styles.button} onPress={showToast}>
        <Text style={styles.buttonText}>Show Toast</Text>
      </TouchableOpacity>

      <Toast ref={(ref) => Toast.setRef(ref)} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f8f8f8',
    paddingHorizontal: 20,
  },
  text: {
    marginBottom: 20,
  },
  pickerContainer: {
    width: '100%',
    marginVertical: 10,
  },
  picker: {
    height: 50,
    width: '100%',
    backgroundColor: '#fff',
    borderRadius: 5,
    borderColor: '#ddd',
    borderWidth: 1,
  },
  label: {
    fontSize: 16,
    marginBottom: 5,
  },
  button: {
    backgroundColor: '#6200ee',
    padding: 15,
    marginTop: 20,
    borderRadius: 8,
    width: '100%',
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
  },
});
