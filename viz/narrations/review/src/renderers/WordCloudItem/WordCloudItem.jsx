// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import React, { useRef, useEffect, useState } from "react";
import { H6, Card, Elevation } from "@blueprintjs/core";
import WordCloud from "./WordCloud";
import "./WordCloud.css";

/*
    EXAMPLE PLUGIN ITEM RENDERER
    Renders mephisto review data items as word clouds of the most common words in the object
    For use inside an ItemListRenderer or AllItemView as an itemRenderer prop
*/

function WordCloudItem({ item }) {
  const SMALL_CARD_WIDTH_LIMIT = 1000;
  const data = item && item.data;
  const id = item && item.id;

  const [cardWidth, setCardWidth] = useState(0);
  const card = useRef();

  useEffect(() => {
    setCardWidth(card.current.offsetWidth);
  }, []);

  const smallCard = cardWidth < SMALL_CARD_WIDTH_LIMIT;

  const normalWordCloudProps = {
    data: data,
    excludedKeys: ["URL"],
    excludedWords: ["true", "false", "the", "with", "on", "in", "of", "and"],
    minFontEmSize: 1,
    maxFontEmSize: 2.5,
    minFontWeight: 100,
    maxFontWeight: 700,
  };

  const smallWordCloudProps = {
    data: data,
    excludedKeys: ["URL"],
    excludedWords: ["true", "false", "the", "with", "on", "in", "of", "and"],
    minFontEmSize: 0.4,
    maxFontEmSize: 1.25,
    minFontWeight: 200,
    maxFontWeight: 600,
  };

  const wordCloudProps = smallCard ? smallWordCloudProps : normalWordCloudProps;

  if (!item) return <p>No Data Available</p>;
  return (
    <>
      <div ref={card} className="word-cloud-item-renderer">
        <Card
          className={
            smallCard
              ? "word-cloud-item-renderer-card small"
              : "word-cloud-item-renderer-card"
          }
          elevation={Elevation.TWO}
          interactive={cardWidth < 1000}
        >
          <H6>
            <b>ID: {id}</b>
          </H6>
          <H6>Data keywords:</H6>
          {/*example WordCloud with example excluded keys and words*/}
          <WordCloud {...wordCloudProps} />
        </Card>
      </div>
    </>
  );
}

export default WordCloudItem;
