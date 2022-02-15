// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import React, { useRef, useEffect, useState } from "react";
import { H6, Card, Elevation } from "@blueprintjs/core";
import "./JSONItem.css";

function JSONItem({ item }) {
  const SMALL_CARD_WIDTH_LIMIT = 1000;
  const [cardWidth, setCardWidth] = useState(0);
  const card = useRef();

  useEffect(() => {
    setCardWidth(card.current.offsetWidth);
  }, []);

  const smallCard = cardWidth < SMALL_CARD_WIDTH_LIMIT;

  return (
    <div
      ref={card}
      className="json-item-renderer"
      id={`item-view-${item && item.id}`}
    >
      <Card
        elevation={Elevation.TWO}
        interactive={smallCard}
        className="json-item-card"
      >
        <pre
          className={
            smallCard
              ? "json-item-renderer-pre small"
              : "json-item-renderer-pre"
          }
        >
          {JSON.stringify(item && item.data)}
        </pre>
        <H6>
          <b>ID: {item && item.id}</b>
        </H6>
      </Card>
    </div>
  );
}

export { JSONItem };
