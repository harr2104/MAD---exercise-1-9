
import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, FlatList } from 'react-native';

const mockDatabase = {
  students: [],
  insert: function (name, rollNo, marks) {
    this.students.push({ name, rollNo, marks });
  },
  update: function (rollNo, newDetails) {
    const index = this.students.findIndex(student => student.rollNo === rollNo);
    if (index !== -1) {
      this.students[index] = { ...this.students[index], ...newDetails };
    }
  },
  delete: function (rollNo) {
    this.students = this.students.filter(student => student.rollNo !== rollNo);
  },
  getAll: function () {
    return this.students;
  },
};

export default function Exp4() {
  const [name, setName] = useState('');
  const [rollNo, setRollNo] = useState('');
  const [marks, setMarks] = useState('');
  const [students, setStudents] = useState(mockDatabase.getAll());

  const handleInsert = () => {
    if (name && rollNo && marks) {
      mockDatabase.insert(name, rollNo, marks);
      setStudents(mockDatabase.getAll());
      clearInputs();
    }
  };

  const handleUpdate = () => {
    if (rollNo && (name || marks)) {
      const updatedDetails = {};
      if (name) updatedDetails.name = name;
      if (marks) updatedDetails.marks = marks;
      mockDatabase.update(rollNo, updatedDetails);
      setStudents(mockDatabase.getAll());
      clearInputs();
    }
  };

  const handleDelete = () => {
    if (rollNo) {
      mockDatabase.delete(rollNo);
      setStudents(mockDatabase.getAll());
      clearInputs();
    }
  };

  const clearInputs = () => {
    setName('');
    setRollNo('');
    setMarks('');
  };

  const renderItem = ({ item }) => (
    <View style={styles.item}>
      <Text>Name: {item.name}</Text>
      <Text>Roll No: {item.rollNo}</Text>
      <Text>Marks: {item.marks}</Text>
    </View>
  );

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.input}
        placeholder="Name"
        value={name}
        onChangeText={setName}
      />
      <TextInput
        style={styles.input}
        placeholder="Roll No"
        value={rollNo}
        onChangeText={setRollNo}
      />
      <TextInput
        style={styles.input}
        placeholder="Marks"
        value={marks}
        onChangeText={setMarks}
      />
      <TouchableOpacity style={styles.button} onPress={handleInsert}>
        <Text style={styles.buttonText}>Insert</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.button} onPress={handleUpdate}>
        <Text style={styles.buttonText}>Update</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.button} onPress={handleDelete}>
        <Text style={styles.buttonText}>Delete</Text>
      </TouchableOpacity>
      <FlatList
        data={students}
        renderItem={renderItem}
        keyExtractor={(item) => item.rollNo}
        style={styles.list}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f8f8f8',
  },
  input: {
    height: 40,
    borderColor: '#ddd',
    borderWidth: 1,
    marginBottom: 10,
    paddingHorizontal: 10,
  },
  button: {
    backgroundColor: '#6200ee',
    padding: 15,
    marginVertical: 5,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
  },
  item: {
    backgroundColor: '#fff',
    padding: 15,
    borderRadius: 5,
    marginVertical: 5,
  },
  list: {
    marginTop: 20,
  },
});
