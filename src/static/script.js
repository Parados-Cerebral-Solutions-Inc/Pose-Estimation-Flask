// Get references to DOM elements
const processedVideo = document.getElementById('processedVideo');
const cameraSelect = document.getElementById('cameraSelect');
const faceBlur = document.getElementById('faceBlur');

// Dynamically connect to the server based on the current hostname
const serverHostAddress = `${window.location.protocol}//${window.location.hostname}:5000`;
const socket = io.connect(serverHostAddress);
let currentStream;

// Get the available cameras and populate the dropdown
async function getCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');

    videoDevices.forEach((device, index) => {
      const option = document.createElement('option');
      option.value = device.deviceId;
      option.text = device.label || `Camera ${index + 1}`;
      cameraSelect.appendChild(option);
    });

    if (videoDevices.length > 0) {
      cameraSelect.value = videoDevices[0].deviceId;
      startVideo(cameraSelect.value);  // Start with the first camera by default
    }
  } catch (err) {
    console.error('Error fetching cameras:', err);
  }
}

// Start capturing frames from the selected camera
async function startVideo(deviceId) {
  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: deviceId ? { exact: deviceId } : undefined
      }
    });

    currentStream = stream;
    const videoTrack = stream.getVideoTracks()[0];
    captureAndSendFrames(videoTrack);
  } catch (err) {
    console.error('Error accessing camera:', err);
  }
}

// Capture frames and send them to the server
function captureAndSendFrames(videoTrack) {
  const imageCapture = new ImageCapture(videoTrack);

  function sendFrame() {
    imageCapture.grabFrame().then(imageBitmap => {
      const canvas = document.createElement('canvas');
      canvas.width = imageBitmap.width;
      canvas.height = imageBitmap.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(imageBitmap, 0, 0);

      const frameData = canvas.toDataURL('image/jpeg');
      socket.emit('frame', frameData);
    }).catch(err => {
      console.error('Error capturing frame:', err);
    });
  }

  setInterval(sendFrame, 100);  // Capture and send frames every 100ms
}

// Listen for the processed frames from the server and update the processedVideo image
socket.on('processed_frame', (data) => {
  processedVideo.src = data;  // Set the processed video frame
});

// Event listener for changing the camera source
cameraSelect.addEventListener('change', () => {
  startVideo(cameraSelect.value);
});

// Monitor the checkbox for enabling/disabling face blur
faceBlur.addEventListener('change', function () {
  const isFaceBlurEnabled = this.checked ? 'enabled' : 'disabled';
  fetch(`/set_face_blur/${isFaceBlurEnabled}`).then(response => {
    if (response.ok) {
      console.log('Face blur updated:', isFaceBlurEnabled);
    }
  });
});

// Initialize camera selection
getCameras();
