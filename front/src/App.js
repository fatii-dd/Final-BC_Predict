import React, { useState } from 'react';
import './App.css';


function App() {
  const [formData, setFormData] = useState({
    BRCA: '',
    weight: '',
    height: '',
    age: '',
    province: '',
    otherProvince: '',
    gender: ''
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { id, value } = e.target;
    setFormData((prev) => ({ ...prev, [id]: value }));
  };

  const handleProvinceChange = (e) => {
    const value = e.target.value;
    setFormData((prev) => ({
      ...prev,
      province: value,
      otherProvince: value === 'อื่นๆ' ? prev.otherProvince : ''
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/predict/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          BRCA: [formData.BRCA],
          BMI_GROUP: { weight: parseFloat(formData.weight), height: parseFloat(formData.height) },
          AGE_GROUP: parseInt(formData.age),
          PROVINCE_GROUP: formData.province === 'อื่นๆ' ? [formData.otherProvince] : [formData.province],
          GENDER_N: formData.gender
        })
      });

      if (!response.ok) throw new Error('Network response was not ok');
      const result = await response.json();
      setResult(result.prediction);
    } catch (error) {
      console.error('Error:', error);
      alert('Error processing your request');
    } finally {
      setLoading(false);
    }
  };

  const getDescription = (result) => {
    switch (result) {
      case 'Healthy':
        return "Based on your input, you are currently assessed as healthy. However, it's important to continue with regular check-ups and maintain a healthy lifestyle.";
      case 'At risk of cancer':
        return "Based on your input, you are assessed to be at risk of cancer. It is strongly advised to consult with a healthcare professional for further examination and guidance.";
      default:
        return "";
    }
  };

  const getResultClass = (result) => {
    switch (result) {
      case 'Healthy':
        return 'result-healthy';
      case 'At risk of cancer':
        return 'result-risk';
      default:
        return '';
    }
  };

  return (
    <div className="container">
      <h2>Self-Assessment for Breast Cancer Risk</h2>
      <form onSubmit={handleSubmit}>
        <label htmlFor="BRCA">BRCA:</label>
        <select id="BRCA" value={formData.BRCA} onChange={handleChange} required>
          <option value="" disabled>กรุณากรอกผลตรวจ BRCA</option>
          <option value="negative">Negative</option>
          <option value="positive">Positive</option>
        </select>

        <label htmlFor="weight">Weight (kg):</label>
        <input type="number" id="weight" value={formData.weight} onChange={handleChange} step="0.1" min="0" required />

        <label htmlFor="height">Height (cm):</label>
        <input type="number" id="height" value={formData.height} onChange={handleChange} step="0.01" min="0" required />

        <label htmlFor="age">Age:</label>
        <input type="number" id="age" value={formData.age} onChange={handleChange} min="0" max="120" required />

        <label htmlFor="province">Province:</label>
        <select id="province" value={formData.province} onChange={handleProvinceChange} required>
          <option value="" disabled>กรุณากรอกจังหวัดที่อยู่อาศัย</option>
          <option value="ยะลา">ยะลา</option>
          <option value="ปัตตานี">ปัตตานี</option>
          <option value="นราธิวาส">นราธิวาส</option>
          <option value="สงขลา">สงขลา</option>
          <option value="สตูล">สตูล</option>
          <option value="พังงา">พังงา</option>
          <option value="พัทลุง">พัทลุง</option>
          <option value="อื่นๆ">อื่นๆ</option>
        </select>

        {formData.province === 'อื่นๆ' && (
          <>
            <label htmlFor="otherProvince">Other Province:</label>
            <input type="text" id="otherProvince" value={formData.otherProvince} onChange={handleChange} required={formData.province === 'อื่นๆ'} />
            </>
        )}

        <label htmlFor="gender">Gender:</label>
        <select id="gender" value={formData.gender} onChange={handleChange} required>
          <option value="" disabled>กรุณาเลือกเพศ</option>
          <option value="Male">Male</option>
          <option value="Female">Female</option>
        </select>

        <button type="submit">Predict</button>
        {loading && <div className="loading"></div>}
      </form>

      {result && (
        <div className="modal" style={{ display: 'flex' }}>
          <div className={`modal-content ${getResultClass(result)}`}>
            <span className="close" onClick={() => setResult(null)}>&times;</span>
            <p className="result-text">{result}</p>
            <p className="description">{getDescription(result)}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
