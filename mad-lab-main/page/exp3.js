import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet } from 'react-native';

export default function Exp3() {
  const [input, setInput] = useState('');
  const [result, setResult] = useState('');

  const handlePress = (value) => {
    if (value === '=') {
      try {
        setResult(eval(input));
      } catch (error) {
        setResult('Error');
      }
    } else if (value === 'C') {
      setInput('');
      setResult('');
    } else if (value === 'sqrt') {
      setInput(`Math.sqrt(${input})`);
    } else if (value === 'pow') {
      setInput(`Math.pow(${input})`);
    } else if (value === 'log') {
      setInput(`Math.log(${input})`);
    } else if (value === 'ln') {
      setInput(`Math.log(${input})`);
    } else if (value === 'cos') {
      setInput(`Math.cos(${input})`);
    } else if (value === 'sin') {
      setInput(`Math.sin(${input})`);
    } else if (value === 'tan') {
      setInput(`Math.tan(${input})`);
    } else if (value === 'mod') {
      setInput(`${input} %`);
    } else {
      setInput(input + value);
    }
  };

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.input}
        value={input}
        onChangeText={setInput}
        placeholder="Enter expression"
        keyboardType="numeric"
      />
      <Text style={styles.result}>{result}</Text>
      <View style={styles.row}>
          <Button title="=" onPress={() => handlePress('=')} />
        </View>
      <View style={styles.buttonContainer}>
        <View style={styles.row}>
          <Button title="C" onPress={() => handlePress('C')} />
          <Button title="+" onPress={() => handlePress('+')} />
          <Button title="*" onPress={() => handlePress('*')} />
          <Button title="/" onPress={() => handlePress('/')} />
        </View>
        <View style={styles.row}>
          <Button title="cos" onPress={() => handlePress('cos')} />
          <Button title="sin" onPress={() => handlePress('sin')} />
          <Button title="tan" onPress={() => handlePress('tan')} />
          <Button title="sqrt" onPress={() => handlePress('sqrt')} />
        </View>
        <View style={styles.row}>
          <Button title="pow" onPress={() => handlePress('pow')} />
          <Button title="log" onPress={() => handlePress('log')} />
          <Button title="ln" onPress={() => handlePress('ln')} />
          <Button title="mod" onPress={() => handlePress('mod')} />
        </View>
      </View>
    </View>
  );
}

const Button = ({ title, onPress }) => (
  <TouchableOpacity style={styles.button} onPress={onPress}>
    <Text style={styles.buttonText}>{title}</Text>
  </TouchableOpacity>
);

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f8f8f8',
    paddingHorizontal: 20,
  },
  input: {
    width: '100%',
    padding: 15,
    marginVertical: 10,
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    fontSize: 18,
    backgroundColor: '#fff',
  },
  result: {
    fontSize: 24,
    marginVertical: 10,
    color: '#333',
  },
  buttonContainer: {
    width: '100%',
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginVertical: 5,
  },
  button: {
    backgroundColor: '#6200ee',
    padding: 15,
    borderRadius: 8,
    width: '22%',
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
  },
});
