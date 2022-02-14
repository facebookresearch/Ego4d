// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import React from "react";
import NarrationsApp from "./NarrationsApp";

function NarrationsItem({ item }) {
  const data = item.data;
  const payload = data.info.payload;

  return (
    <div className="json-item-renderer">
      <NarrationsApp data={data} />
    </div>
  );
}

export default NarrationsItem;
