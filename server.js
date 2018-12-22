// Include the sa module, needs installing first (npm install sentiment-analysis)
const sentimentAnalysis = require('sentiment-analysis');

function displaySentiment(inputText, sentimentScore){

 // Determine if the score is positive, neutral or negative
  let overallSentiment = 'neutral';
  if (sentimentScore > 0.2) overallSentiment = 'positive';
  else if (sentimentScore < -0.2) overallSentiment = 'negative';

  // Get persnt from score
  const persentageScore = `${sentimentScore * 100}%`;

  // This is the sentence to return (e.g. 40% positive)
  const sentence = `${persentageScore} (${overallSentiment})`;

  return sentence;
}
/**
* Returns a color to represent the sentimentScore
* Either green, yellow or red. In the console format
*/
function getChalkColor(sentimentScore){
  let chalkColor = '\x1b[33m';
  if(sentimentScore > 0.2) chalkColor = '\x1b[32m';
  else if(sentimentScore < -0.2) chalkColor = '\x1b[31m';
  return chalkColor;
}

var readline = require('readline');
var rl = readline.createInterface({
  input: process.stdin,
  output: '',
  terminal: false
});

function findsentiment(line){
  const sentiment = sentimentAnalysis(line); // Get the senitment score
  const color = getChalkColor(sentiment); // Get a colour to represent score
  return displaySentiment(line, sentiment);
}
  
  // Actuall print the results to the console (in color!)
  //console.log(color, displaySentiment(line, sentiment), resetColor, '\t'+text);


var fs = require('fs');

function readLines(input, func) {
  var remaining = '';

  input.on('data', function(data) {
    remaining += data;
    //const sentiment= findsentiment(data);

    var index = remaining.indexOf('\n');
    while (index > -1) {

      var line = remaining.substring(0, index);
      remaining = remaining.substring(index + 1);
      func(line);
      index = remaining.indexOf('\n');
    }
  });

  input.on('end', function() {
    if (remaining.length > 0) {
      func(remaining);
    }
  });
}

function func(line) {
	
	const sentiment = sentimentAnalysis(line); // Get the senitment score
	const color = getChalkColor(sentiment); // Get a colour to represent score
	var fs = require('fs');

	var data = displaySentiment(line, sentiment);
	data+='\n';
	fs.appendFile('output2.txt',data, 'utf8',
    // callback function
    function(err) { 
        if (err) throw err;
        // if no error
        console.log("Data is appended to file successfully.")
	});
	//console.log('Line: ' + line);
	//console.log(displaySentiment(line, sentiment));


}

var input = fs.createReadStream('output.txt');
readLines(input, func);




//rl.end(function (err) {
//    if (err){
//        throw err;
//    };

//    console.log('finished');
//});
/*
var express = require('express');
var app = express();
//Middleware
app.listen(3005)

// Creates a server which runs on port 3000 and
// can be accessed through localhost:3000
// Use python shell

// end the input stream and allow the process to exit

// Function callName() is executed whenever
// the URL is of the form localhost:3000/name
app.get('/name', callName);

function callName(req, res) {

    // Use child_process.spawn method from
    // child_process module and assign it
    // to variable spawn
    var spawn = require("child_process").spawn;

    // Parameters passed in spawn -
    // 1. type_of_script
    // 2. List containing Path of the script
    //    and arguments for the script

    // E.g.: http://localhost:3000/name?firstname=Mike&lastname=Will
    // So, first name = Mike and last name = Will
    var process = spawn('python',["sentiment1.py"] );

    // Takes stdout data from script which executed
    // with arguments and send this data to res object
    process.stdout.on('data', function(data) {
        res.send(data.toString());
    } )

    process.stdin.write(JSON.stringify(data));
	process.stdin.end();
	console.log(data);
}
*/
// Array of example text, to calculate sentiments from
