#include <ADC.h>
#include <Servo.h> 
 
Servo servoBend;
Servo servoRot; 
 
int posBend = 69;   
int posRot = 90;
int posMove = 0;
int posMovebefore = 0;

//Initial Positions, start positions and reset positions
const int posBendHomeLow = 69; 
const int posRotHomeLow = 90;

const int posBendHomeHigh =1255;
const int posRotHomeHigh =1472;


//Mode - Reolution: 0: low - value from 0-400, 1: high - values 400+
int bendMode  = 1; 
int rotMode = 1;



// define pin for linear stepper
const int EN = 7;     
const int M0 = 6;      
const int M1 = 5;
const int M2 = 4;
const int STDBY = 3;
const int STEP = 2;
const int DIR = 1;



void home(){

  if(bendMode ==0){
    posBend = posBendHomeLow;
  }else{
    posBend=posBendHomeHigh;
  }
  if(rotMode ==0){
    posRot = posRotHomeLow;
  }else{
    posRot=posRotHomeHigh;
  }

  delay(100);
  servoBend.write(posBend); 
  delay(200);
  servoRot.write(posRot); 
  
  delay(200);

  digitalWrite(DIR, LOW);
  for(int counter = 0;counter <= posMove;counter++) {
    digitalWrite(STEP, HIGH);
    delayMicroseconds(700);
    digitalWrite(STEP, LOW);
    delayMicroseconds(700);
  }
  posMove = 0;
  posMovebefore = posMove;
  delay(20);
}

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
  home();

}




void loop()
{
  if (Serial.available() > 0)
  {
    String command = Serial.readStringUntil('\n');
    if (command.length() > 1) {
      char c = command.charAt(0);  // Extract the letter
      int number = command.substring(1).toInt();  // Extract the number and convert to integer
            
      switch (c) {
        case 't': // move
          if ((number >= 0) && (number <= 3600)){
            posMove = number;
            int numsteps = abs(posMove - posMovebefore);
            if(posMove > posMovebefore){
              digitalWrite(DIR, HIGH);
            }
            else{
              digitalWrite(DIR, LOW);
            }
            for(int counter = 0;counter <= numsteps;counter++) {
              digitalWrite(STEP, HIGH);
              delayMicroseconds(700);
              digitalWrite(STEP, LOW);
              delayMicroseconds(700);
            }
            Serial.println(posMove);
            posMovebefore = posMove;
            //delay(20);
          }
          break;
        case 'b': // bend 
          if (number < 400){
            if ((number >= 40) && (number <= 98)){
              posBend = number;
              servoBend.write(posBend); 
              Serial.println(posBend);
              delay(20);
            }
            bendMode=0;
          }
          else{
            if ((number >= 956) && (number <= 1554)){
              posBend = number;
              servoBend.write(posBend); 
              Serial.println(posBend);
              delay(20);
            }
            else{
              Serial.println(posBend);
            }
            bendMode=1;
          }

          break;
        case 'r': // rotate 
          if (number < 400){
            if ((number >= 0) && (number <= 180)){
              posRot = number;
              servoRot.write(posRot); 
              Serial.println(posRot);
              delay(20);
            }
            rotMode=0;
          }
          else{
            if ((number >= 544) && (number <= 2400)){
              posRot = number;
              servoRot.write(posRot); 
              Serial.println(posRot);
              delay(20);
            }
            else{
              Serial.println(posRot);
            }
            rotMode=1;
          }
          break;
      }
    }
    else{
      char c = command.charAt(0);;
      switch (c) {
        case 'f': // move forward
          if (posMove < 3600){
            posMove += 10;
            digitalWrite(DIR, HIGH);
            //delayMicroseconds(700);
            digitalWrite(STEP, HIGH);
            delayMicroseconds(700);
            digitalWrite(STEP, LOW);
            delayMicroseconds(700);
            Serial.println(posMove);
          }
          break;

        case 'b': // move backward
          if (posMove >= 20){
            posMove -= 10;
            digitalWrite(DIR, LOW);
            //delayMicroseconds(700);
            digitalWrite(STEP, HIGH);
            delayMicroseconds(700);
            digitalWrite(STEP, LOW);
            delayMicroseconds(700);
            Serial.println(posMove);
          }
          break;

        case 'u': // bend up
          if (posBend < 96){
            posBend += 1;
            servoBend.write(posBend); 
            Serial.println(posBend);
            delay(15);
            
          }
          bendMode=0;
          break;
        case 'd': // bend down
          if (posBend >= 42){
            posBend -= 1;
            servoBend.write(posBend); 
            Serial.println(posBend);
            delay(15);
          }
          bendMode=0;
          break;
        case 'r': // rotate right
          if (posRot >= 2){
            posRot += 1;
            servoRot.write(posRot);
            Serial.println(posRot);
            delay(15);
          }
          rotMode=0;
          break;
        case 'l': // rotate left 
          if (posRot < 180){
            posRot -= 1;
            servoRot.write(posRot); 
            Serial.println(posRot);
            delay(15);
          }
          rotMode=0;
          break;
        case 'i': // initialize 
          home();

          break;

        case 'j': //return the current values of the joints
          int m_jointB = servoBend.readMicroseconds();
          int m_jointR = servoRot.readMicroseconds();
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
}