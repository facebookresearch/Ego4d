// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import React from "react";
import { H6, Card, Elevation } from "@blueprintjs/core";
import { getHostname } from "../utils";
import { getAllObjectValWordCounts } from "../renderers/WordCloudItem/WordCloud";

function NarrationsThumbnail({ item }) {
  const data = item.data;
  const payload = data.info.payload;

  const [isError, setError] = React.useState(false);

  return (
    <div className="json-item-renderer">
      <Card
        elevation={Elevation.TWO}
        interactive={true}
        className="json-item-card"
      >
        <p style={{ fontSize: 12 }}>
          {data.uid} &mdash; {payload.length} entries
        </p>
        <img
          role="presentation"
          onError={(e) => {
            // console.log(e);
            e.target.onerror = null;
            // e.target.src = "image_path_here";
            setError(true);
          }}
          src={
            data.img && !isError
              ? getHostname() + data.img
              : "data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="
          }
          alt="Thumbnail"
          style={{ width: data.img ? "100%" : "1px" }}
        />
        {/* {JSON.stringify(payload[0].label)} */}
        <p>
          {getKeyWords(payload).map((word) => (
            <span
              style={{
                marginRight: "1em",
                fontStyle: "italic",
                display: "inline-block",
              }}
              key={word}
            >
              #{word}
            </span>
          ))}
        </p>
      </Card>
    </div>
  );
}

function getKeyWords(payload) {
  const counts = getAllObjectValWordCounts(
    payload,
    [
      "C",
      "the",
      "be",
      "of",
      "from",
      "to",
      "and",
      "a",
      "in",
      "that",
      "have",
      "it",
      "for",
      "not",
      "on",
      "with",
      "by",
      "his",
      "her",
      "up",
      "down",
    ],
    ["id"]
  );
  const mostCommonWords = Object.entries(counts).sort(
    ([firstKey, firstValue], [secondKey, secondValue]) =>
      secondValue - firstValue
  );
  return mostCommonWords.slice(0, 10).map(([word, _count]) => word);
}

export default NarrationsThumbnail;
