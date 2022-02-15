// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import React from "react";
import "./WordCloud.css";

//returns an object with a count for all words seperated by spaces in a string, punctuation is removed
const getAllStringWordCounts = (str, excludedWords = []) => {
  //object to be indexed by string words mapped to integer word counts
  var vals = {};
  //separate string into array of space separated words, excluded words are ignored
  var words = str.split(" ").filter((word) => {
    if (!word || word === "" || word.length === 1) return false;
    if (!isNaN(parseFloat(str))) return false;
    for (const char of excludedWords) {
      if (word.toLowerCase().indexOf(char.toLowerCase()) > -1) return false;
    }
    return true;
  });
  //remove punctuation from all words
  words = words.map((word) => word.replace(/[.,/#!$%^&*;:{}=\-_`~()]/g, ""));
  //count words
  for (let word of words) {
    if (word in vals) {
      vals[word] += 1;
    } else {
      vals[word] = 1;
    }
  }
  return vals;
};

//recursively iterates the values of an object and counts the occurence of space separated words in string values
const getAllObjectValWordCounts = (
  dict,
  excludedWords = [],
  excludedKeys = []
) => {
  //object to be indexed by string words mapped to integer word counts
  let vals = {};
  //get all keys for an object, ignoring excluded keys
  const keys = Object.keys(dict).filter((key) => {
    for (const filter of excludedKeys) {
      if (key === filter) return false;
    }
    return true;
  });

  for (let key of keys) {
    let val = dict[key];
    //if value of key is empty skip
    if (!val || val === "") continue;
    if (typeof val === "object") {
      //if value is object, recurse into the object
      const recursedVals = getAllObjectValWordCounts(
        val,
        excludedWords,
        excludedKeys
      );
      //get keys from recursed word counts
      const recursedKeys = Object.keys(recursedVals);
      //merge word counts into parent count
      for (let rKey of recursedKeys) {
        if (rKey in vals) {
          vals[rKey] = recursedVals[rKey] + vals[rKey];
        } else {
          vals[rKey] = recursedVals[rKey];
        }
      }
    } else if (typeof val === "string") {
      //if value of key is string, get all word counts of string
      const strData = String(val);
      const strCounts = getAllStringWordCounts(strData, excludedWords);
      //get keys from word count
      const strKeys = Object.keys(strCounts);
      //merge word counts into parent count
      for (let sKey of strKeys) {
        if (sKey in vals) {
          vals[sKey] = strCounts[sKey] + vals[sKey];
        } else {
          vals[sKey] = strCounts[sKey];
        }
      }
    }
  }
  return vals;
};

/*
  For a given object or string, generates a paragraph filled with different sized and weighted spans of words based on the frequency of occurrence in the object or string
  More frequently occurring words will be sized larger with larger font weighting as well
  The object or string to be parsed must be passed in a property named 'data'
  Objects will be traversed recursively for all attributes in string form, all strings in the object or passed in through params will be parsed for space separated words.
  Words can be excluded by passing a property named "excludedWords" which must be an array of strings
  Keys in a object can be excluded by passing a property named "excludedKeys" which must be an array of strings
  Maximum font size in EM units can be set by passing a maxFontEmSize property
  Minimum font size in EM units can be set by passing a minFontEmSize property
  Maximum font weight can be set by passing a maxFontWeight property
  Minimum font weight can be set by passing a minFontWeight property
*/
function WordCloud({
  data,
  minFontEmSize = 0.4,
  maxFontEmSize = 1.25,
  minFontWeight = 200,
  maxFontWeight = 600,
  excludedWords = [],
  excludedKeys = [],
  style,
}) {
  if (!data || data === {}) return <p>No Data Available</p>;

  //maps number from given range to a new range
  const scaleNumber = (num, inMin, inMax, outMin, outMax, round) => {
    return round === true
      ? Math.ceil(
          ((num - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin
        )
      : ((num - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
  };

  //reduces deviations between word counts given input of an object representing string words mapped to word count integers
  // and maps word counts to em font sizes
  const scaleEntries = (wordDict) => {
    //get all words recorded
    var words = Object.keys(wordDict);
    //if empty, abort
    if (!words || words.length === 0) return {};
    //find min and max of all words
    var values = Object.values(wordDict);
    var maxVal = Math.max.apply(Math, values);
    var minVal = Math.min.apply(Math, values);

    if (maxVal - minVal > 10) {
      //if difference between max and min word counts are over 10 remove all min counts
      for (const word of words) {
        if (wordDict[word] === minVal) delete wordDict[word];
      }

      //recalculate words, values, min and max
      words = Object.keys(wordDict);
      values = Object.values(wordDict);
      maxVal = Math.max.apply(Math, values);
      minVal = Math.min.apply(Math, values);
    }

    //calculate average word count
    var sum = 0;
    for (const word of words) {
      sum += wordDict[word];
    }
    const avg = Math.ceil(sum / words.length);

    //set maximum accepted word count to difference between average and smallest value in list
    const diff = avg - minVal;
    const maxRange = avg + diff;

    //reduce all values over maximum accepted word count to the maximum accepted word count
    for (const word of words) {
      wordDict[word] = Math.min(wordDict[word], maxRange);
    }

    //maps word counts to em font sizes
    for (const word of words) {
      wordDict[word] = scaleNumber(
        wordDict[word],
        1,
        maxRange,
        minFontEmSize,
        maxFontEmSize
      );
    }
    return wordDict;
  };

  //get all word counts
  var wordCounts = {};
  switch (typeof data) {
    case "object":
      wordCounts = getAllObjectValWordCounts(data, excludedWords, excludedKeys);
      break;
    case "string":
      wordCounts = getAllStringWordCounts(data, excludedWords);
      break;
    default:
      break;
  }

  //normalize word counts and map to em font sizes
  wordCounts = scaleEntries(wordCounts);
  const wordKeys = Object.keys(wordCounts);

  //replace em font sizes in list with css styles for spans
  for (const word of wordKeys) {
    const size = wordCounts[word];
    wordCounts[word] = {
      fontSize: size + "em",
      fontWeight: scaleNumber(
        size,
        minFontEmSize,
        maxFontEmSize,
        minFontWeight,
        maxFontWeight,
        true
      ),
    };
  }

  //map css styles to spans with corresponding word and populate paragraph with spans
  return (
    <div style={style} className="word-cloud">
      <p>
        {wordKeys.map((word) => (
          <span style={wordCounts[word]}>{" " + word + " "}</span>
        ))}
      </p>
    </div>
  );
}

export default WordCloud;

export { getAllObjectValWordCounts, getAllStringWordCounts };
