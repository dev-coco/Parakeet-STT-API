# Parakeet-STT-API


```JavaScript
const formData = new FormData()
formData.append('audio', blob)
const response = await fetch('http://localhost:1643/transcribe', {
  method: 'POST',
  body: formData
})
const json = await response.json()
console.log(json)
```
