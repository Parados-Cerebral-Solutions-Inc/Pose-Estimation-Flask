// Get list of cameras and populate the dropdown
async function getCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const videoDevices = devices.filter((device) => device.kind === 'videoinput');

  const select = document.getElementById('cameraSelect');
  videoDevices.forEach((device, index) => {
    const option = document.createElement('option');
    option.value = index;
    option.text = device.label || `Camera ${index + 1}`;
    select.appendChild(option);
  });

  // Add event listener to change camera source
  select.addEventListener('change', function () {
    const cameraIndex = select.value;
    fetch(`/set_camera/${cameraIndex}`).then((response) => {
      if (response.ok) {
        console.log('Camera source updated');
      }
    });
  });
}

// Monitor checkbox state and update server
document.getElementById('faceBlur').addEventListener('change', function () {
  const isFaceBlurEnabled = this.checked ? 'enabled' : 'disabled';
  fetch(`/set_face_blur/${isFaceBlurEnabled}`).then((response) => {
    if (response.ok) {
      console.log('Face blur updated:', isFaceBlurEnabled);
    }
  });
});

// Call the function to populate the camera list
getCameras();
