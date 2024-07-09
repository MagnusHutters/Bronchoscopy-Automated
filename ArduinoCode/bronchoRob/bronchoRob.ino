/*
Broncho Robot controller, SDU
written by Zhuoqi Cheng 19-06-2024
*/

#include <Servo.h> 

// define 2 servo for bending and rotation control
Servo servoBend;
Servo servoRot; 

// define pin for linear stepper
const int EN = 7;     
const int M0 = 6;      
const int M1 = 5;
const int M2 = 4;
const int STDBY = 3;
const int STEP = 2;
const int DIR = 1;
 
// define variables for 3 DoFs
float posBend = 69;  
float posRot = 90;
float posMove = 0;
int intPosBend = 69;  
int intPosRot = 90;
int intPosMove = 0;

void setup()
{
  servoBend.attach(8);
  delay(1);
  servoRot.attach(9);
  delay(1);
  analogWriteResolution(12);
  delay(1);
  Serial.begin(115200);
  delay(1);
  pinMode(STDBY, OUTPUT);
  digitalWrite(STDBY, HIGH); //device in stand-by
  pinMode(M0, OUTPUT );
  pinMode(M1, OUTPUT );
  pinMode(M2, OUTPUT);
  pinMode(DIR, OUTPUT);digitalWrite(DIR, LOW);
  pinMode(STEP, OUTPUT);
  pinMode(EN, OUTPUT); digitalWrite(EN, HIGH); // output stage disabled
  digitalWrite(M0, LOW);digitalWrite(M1, HIGH);digitalWrite(M2, LOW); //initial mode configuration -- this can be changed to diff resolution 
  // initialization
  servoBend.write(posBend); 
  delay(100);
  servoRot.write(posRot); 
  delay(100);
}


void loop()
{
  if (Serial.available() > 0)
  {
    char c = Serial.read();
    switch (c) {
      case 'f': // move forward
        for(int counter = 0;counter < 4;counter++) {
          if (posMove < 3600){
            posMove += 1;
            digitalWrite(DIR, HIGH);
            //delayMicroseconds(2);
            digitalWrite(STEP, HIGH);
            delayMicroseconds(700);
            digitalWrite(STEP, LOW);
            delayMicroseconds(700);
          }
        }
        Serial.println((int)posMove);
        break;
      case 'b': // move backward
        

        for(int counter = 0;counter < 4;counter++) {
          if (posMove >= 20){
            posMove -= 1;
            digitalWrite(DIR, LOW);
            //delayMicroseconds(2);
            digitalWrite(STEP, HIGH);
            delayMicroseconds(700);
            digitalWrite(STEP, LOW);
            delayMicroseconds(700);
          }
        }
        Serial.println((int)posMove);
        break;
      case 'u': // bend up - added 1 to account for slack
        if (posBend-1 < 96){
          posBend += 0.25f;

          intPosBend=(int)ceil(posBend)+1;
          servoBend.write(intPosBend); 
          Serial.println(intPosBend);
          delay(30);
        }
        break;
      case 'd': // bend down - subtracted 1 to account for slack
        if (posBend+1 >= 42){
          posBend -= 0.25f;
          intPosBend=(int)floor(posBend)-1;
          servoBend.write(intPosBend); 
          Serial.println(intPosBend);
          delay(30);
        }
        break;
      case 'r': // rotate right - clockwise & the image counter-clockwise 
        if (posRot < 180){
          posRot += 1;
          intPosRot=(int)round(posRot);

          servoRot.write(intPosRot); 
          Serial.println(intPosRot);
          delay(30);
        }
        break;
      case 'l': // rotate left - counter-clockwise and the image clockwise
        if (posRot >= 2){
          posRot -= 1;
          intPosRot=(int)round(posRot);
          servoRot.write(intPosRot);
          Serial.println(intPosRot);
          delay(30);
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

        digitalWrite(DIR, LOW);
        for(int counter = 0;counter <= posMove;counter++) {
          digitalWrite(STEP, HIGH);
          delayMicroseconds(700);
          digitalWrite(STEP, LOW);
          delayMicroseconds(700);
        }
        posMove = 0;
        //posMovebefore = posMove;
        delay(20);

        break;
      case 'j': //return the current values of the joints
        int m_jointB = servoBend.read();
        int m_jointR = servoRot.read();
        int m_jointT = posMove;
        //int[] values = {m_jointB, m_jointR, m_jointT};
        Serial.println(m_jointB);
        delay(15);
        Serial.println(m_jointR);
        delay(15);
        Serial.println(m_jointT);
        //Serial.println(0);
        delay(15);
        break;
    }
  }
}


/*
String command = Serial.readStringUntil('\n');
    if (command.length() > 1) {
      c = command.charAt(0);  // Extract the letter
      number = command.substring(1).toInt();  // Extract the number and convert to integer
            
      switch (c) {
        case 'm': // move
          if ((number >= 0) && (number <= 3600)){
            posMove = number;
            analogWrite(A22, posMove);
            delay(20);
          }
          break;
        case 'b': // bend 
          if ((number >= 0) && (number <= 170)){
            posBend = number;
            servoBend.write(posBend); 
            delay(20);
          }
          break;
        case 'r': // rotate 
          if ((number >= 0) && (number <= 170)){
            posRot = number;
            servoRot.write(posRot); 
            delay(20);
          }
          break;
        case 'i': // initialize 
          posBend = 0;   
          posRot = 0;
          posMove = 0;
          analogWrite(A22, posMove);
          delay(100);
          servoBend.write(posBend); 
          delay(100);
          servoRot.write(posRot); 
          delay(100);
          break;
      }
    }
    */