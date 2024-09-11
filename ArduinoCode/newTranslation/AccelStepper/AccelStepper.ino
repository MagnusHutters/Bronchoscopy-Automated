#include <ADC.h>
#include <Servo.h> 
#include <TeensyThreads.h>
#include <AccelStepper.h>

Servo servoBend;
Servo servoRot; 
 
int posBend = 69;   
int posRot = 90;
int posMovebefore = 0;
int posMove = 0;


// define pin for linear stepper
const int EN = 7;     
const int M0 = 6;      //M1 datasheet
const int M1 = 5;      //M2 datasheet
const int M2 = 4;      //M3 datasheet
const int STDBY = 3; 
const int STEP = 2; //stepPin
const int DIR = 1; //dirPin



//int posMovebefore = 0;
unsigned long currentMillis = millis();
unsigned long previousMillis = millis();

#define motorInterfaceType 1

// Create a new instance of the AccelStepper class:
AccelStepper stepper = AccelStepper(AccelStepper::DRIVER, STEP, DIR);
volatile int m_signal = 1;
volatile int m_vel = 400 * 64;
bool m_keyboardmove = false;

void setup()
{
  servoBend.attach(8); // arduino pin where the servo is attatched
  delay(1);
  servoRot.attach(10);
  delay(1);
  analogWriteResolution(12);
  delay(1);
  Serial.begin(115200);
  delay(1);

  // Linear motor
  //pinMode(STDBY, OUTPUT);
  digitalWrite(STDBY, HIGH); //device in stand-by
  pinMode(M0, OUTPUT );
  pinMode(M1, OUTPUT );
  pinMode(M2, OUTPUT);
  //pinMode(DIR, OUTPUT);digitalWrite(DIR, LOW);
  //pinMode(STEP, OUTPUT);
  //pinMode(EN, OUTPUT); digitalWrite(EN, HIGH); // output stage disabled
  digitalWrite(M0, HIGH);digitalWrite(M1, HIGH);digitalWrite(M2, HIGH); //initial mode configuration -- this can be changed to diff resolution 

  // initialization
  servoBend.write(posBend); 
  delay(100);
  servoRot.write(posRot); 
  delay(100);

  // Set the maximum speed in steps per second:
  stepper.setEnablePin(EN);
  stepper.setMaxSpeed(1000 * 64); //1000
  //stepper.setMinPulseWidth(10000);
  //stepper.setAcceleration(30);
  

}


void loop()
{

  posMovebefore = stepper.currentPosition();
  if(stepper.currentPosition() != posMove && !m_keyboardmove)
  {
    stepper.setSpeed(m_signal * m_vel);
    stepper.runSpeed(); 
  }

  if(m_keyboardmove){
    if(((posMovebefore >= 0) && (posMovebefore <= 3600 * 64))){
      posMove = posMovebefore;
      stepper.setSpeed(m_signal * m_vel);
      stepper.runSpeed(); 
    }
  }

  if (Serial.available() > 0)
  {
    String command = Serial.readStringUntil('\n');
    if (command.length() > 1) {
      char c = command.charAt(0);  // Extract the letter
      int number = command.substring(1).toInt();  // Extract the number and convert to integer
            
      switch (c) {
        case 't': // move
          if ((number >= 0) && (number <= 3600 * 64)){
            posMove = number;
            m_signal = 1;

            if (posMove < stepper.currentPosition()){
              m_signal = -1;
            }

            Serial.println(posMovebefore);
          }


          break;
          
        case 'b': // bend 
          if (number < 400){
            if ((number >= 40) && (number <= 98)){
              posBend = number;
              servoBend.write(posBend); 
              Serial.println(posBend);
              //delay(20);
            }
          }
          else{
            if ((number >= 850) && (number <= 1700)){
              posBend = number;
              servoBend.write(posBend); 
              Serial.println(posBend);
              //delay(20);
            }
            else{
              Serial.println(posBend);
            }
          }

          break;
        case 'r': // rotate 
          if (number < 400){
            if ((number >= 0) && (number <= 180)){
              posRot = number;
              servoRot.write(posRot); 
              Serial.println(posRot);
              //delay(20);
            }
          }
          else{
            if ((number >= 544) && (number <= 2400)){
              posRot = number;
              servoRot.write(posRot); 
              Serial.println(posRot);
              //delay(20);
            }
            else{
              Serial.println(posRot);
            }
          }
          break;
      }
    }
    else{
      char c = command.charAt(0);;
      switch (c) {
        case 'f': // move forward
          if (posMove <= 3580 * 64){
            m_signal = 1;
            m_keyboardmove = true;
          }
          Serial.println(posMovebefore);
          //delay(20);
          break;

        case 'b': // move backward
          if (posMove >= 20){
            m_signal = -1;
            m_keyboardmove = true;
          }
          Serial.println(posMovebefore);
          //delay(20);
          break;
        case 's': // release of the keyboard movement
          m_keyboardmove = false;
          m_signal = 1;

          posMovebefore = stepper.currentPosition();
          posMove = posMovebefore;
          if(posMove <= 0){
            posMove = 0;
          }
          if(posMove >= 3600* 64){
            posMove = 3600 * 64;
          }    
          Serial.println(posMovebefore);
          //delay(20);
          break;
        case 'u': // bend up
          if (posBend < 96){
            posBend += 1;
            servoBend.write(posBend); 
            Serial.println(posBend);
            delay(15);
          }
          break;
        case 'd': // bend down
          if (posBend >= 42){
            posBend -= 1;
            servoBend.write(posBend); 
            Serial.println(posBend);
            delay(15);
          }
          break;
        case 'l': // rotate left 
          if (posRot < 180){
            posRot += 1;
            servoRot.write(posRot); 
            Serial.println(posRot);
            delay(15);
          }
          break;
        case 'r': // rotate right
          if (posRot >= 2){
            posRot -= 1;
            servoRot.write(posRot);
            Serial.println(posRot);
            delay(15);
          }
          break;
        case 'i': // initialize 
          posBend = 69;   
          posRot = 90;

          delay(100);
          servoBend.write(posBend); 
          delay(100);
          servoRot.write(posRot); 
          delay(100);

          posMove = 0;
          if (posMove < stepper.currentPosition()){
              m_signal = -1;
          }
          while(stepper.currentPosition() != posMove)
          {
            stepper.setSpeed(m_signal * m_vel);
            stepper.runSpeed();
          }
          posMovebefore = posMove;
          delay(20);

          break;

        case 'j': //return the current values of the joints
          int m_jointB = servoBend.readMicroseconds();
          int m_jointR = servoRot.readMicroseconds();
          int m_jointT = posMovebefore;
          //int[] values = {m_jointB, m_jointR, m_jointT};
          Serial.println(m_jointB);
          //delay(15);
          Serial.println(m_jointR);
          //delay(15);
          Serial.println(m_jointT);
          //Serial.println(0);
          //delay(15);
          break;
      }
    }
  }
}