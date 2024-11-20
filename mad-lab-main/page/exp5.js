// app/exp5.js
import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, Alert } from 'react-native';

export default function Exp5() {
  const [username, setUsername] = useState('');
  const [id, setId] = useState('');

  const validateInputs = () => {
    if (!username || !id) {
      Alert.alert('Error', 'Both fields are required.');
      return;
    }

    const nameRegex = /^[A-Za-z]+$/;
    const idRegex = /^\d{4}$/;

    if (!nameRegex.test(username)) {
      Alert.alert('Error', 'Username must contain only alphabets.');
      return;
    }

    if (!idRegex.test(id)) {
      Alert.alert('Error', 'ID must be a 4-digit numeric value.');
      return;
    }

    Alert.alert('Success', 'Inputs are valid!');
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>User Validation</Text>
      
      <TextInput
        style={styles.input}
        placeholder="Username"
        value={username}
        onChangeText={setUsername}
        autoCapitalize="words"
      />

      <TextInput
        style={styles.input}
        placeholder="ID"
        value={id}
        onChangeText={setId}
        keyboardType="numeric"
        maxLength={4}
      />

      <TouchableOpacity style={styles.button} onPress={validateInputs}>
        <Text style={styles.buttonText}>Validate</Text>
      </TouchableOpacity>
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
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    color: '#333',
  },
  input: {
    width: '100%',
    padding: 15,
    marginVertical: 10,
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    fontSize: 16,
    backgroundColor: '#fff',
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
