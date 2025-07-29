#define RPWM 19
#define LPWM 18
#define R_EN 17
#define L_EN 16
#define L_EN2 25
#define R_EN2 23
#define RPWM2 21
#define LPWM2 22
#define TRIG_PIN 33
#define ECHO_PIN 32

String input;
unsigned long lastCommandTime = 0;
const unsigned long COMMAND_TIMEOUT = 5000; // 1 second timeout

void setup() {
  pinMode(RPWM, OUTPUT);
  pinMode(LPWM, OUTPUT);
  pinMode(R_EN, OUTPUT);
  pinMode(L_EN, OUTPUT);
  pinMode(RPWM2, OUTPUT);
  pinMode(LPWM2, OUTPUT);
  pinMode(R_EN2, OUTPUT);
  pinMode(L_EN2, OUTPUT);
  digitalWrite(R_EN, HIGH);
  digitalWrite(L_EN, HIGH);
  digitalWrite(R_EN2, HIGH);
  digitalWrite(L_EN2, HIGH);
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  
  lastCommandTime = millis(); // Initialize timestamp
}

void loop() {
  // Distance sensor
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long duration = pulseIn(ECHO_PIN, HIGH);
  float distance_cm = duration * 0.0343 / 2.0;
  Serial.print("DIST:");
  Serial.println(distance_cm);
  
  // Serial reading
  input = "";
  bool command_ready = false;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      command_ready = true;
      break;
    } else if (c >= 32 && c <= 126) {
      input += c;
    }
  }
  
  if (command_ready) {
    Serial.println(input); // echo for debug
    
    if (input == "F") {
      analogWrite(LPWM, 255);
      analogWrite(RPWM2, 255);
      analogWrite(RPWM, 0);
      analogWrite(LPWM2, 0);
    } else if (input == "B") {
      analogWrite(RPWM, 255);
      analogWrite(LPWM, 0);
      analogWrite(RPWM2, 0);
      analogWrite(LPWM2, 255);
    } else if (input == "L") {
      analogWrite(RPWM, 0);
      analogWrite(LPWM, 255);
      analogWrite(RPWM2, 0);
      analogWrite(LPWM2, 255);
    } else if (input == "R") {
      analogWrite(RPWM, 255);
      analogWrite(LPWM, 0);
      analogWrite(RPWM2, 255);
      analogWrite(LPWM2, 0);
    } else if (input == "S") {
      stop();
    }
    
    lastCommandTime = millis(); // Update timestamp when command received
  }
  
  // SAFETY: Stop if no command received for 1 second
  if (millis() - lastCommandTime > COMMAND_TIMEOUT) {
    stop();
  }
  
  delay(50); // reduce serial spam
}

void stop() {
  analogWrite(RPWM, 0);
  analogWrite(LPWM, 0);
  analogWrite(RPWM2, 0);
  analogWrite(LPWM2, 0);
}
