// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import React, { useState } from "react";
import { Redirect } from "react-router-dom";
import { useMephistoReview } from "mephisto-review-hook";
import {
  InputGroup,
  Button,
  Navbar,
  NavbarGroup,
  NavbarDivider,
  NavbarHeading,
  Alignment,
} from "@blueprintjs/core";
import { Tooltip } from "@blueprintjs/core";
import { GridCollection, JSONItem } from "../renderers";
import { Pagination } from "./pagination";
import { getHostname } from "../utils";
import ErrorPane from "./ErrorPane";

function CollectionView({
  itemRenderer = JSONItem,
  collectionRenderer: CollectionRenderer = GridCollection,
  pagination = true,
  resultsPerPage = 12,
}) {
  const [page, setPage] = useState(pagination ? 1 : null);
  const [filters, setFilters] = useState("");
  const [filtersBuffer, setFiltersBuffer] = useState("");
  const [filterTimeout, setFilterTimeout] = useState(null);

  const { data, isFinished, isLoading, error, mode, totalPages } =
    useMephistoReview({
      page,
      resultsPerPage,
      filters,
      hostname: getHostname(),
    });

  const setFiltersAndResetPage = (filtersStr) => {
    if (page !== null && page !== 1) setPage(1);
    setFilters(filtersStr);
  };

  const delaySetFilters = (filtersStr) => {
    setFiltersBuffer(filtersStr);
    if (filterTimeout) {
      clearTimeout(filterTimeout);
    }
    setFilterTimeout(
      setTimeout(() => {
        setFiltersAndResetPage(filtersStr);
      }, 3000)
    );
  };

  const setFiltersImmediately = () => {
    if (filterTimeout) {
      clearTimeout(filterTimeout);
    }
    setFiltersAndResetPage(filtersBuffer);
  };

  const searchButton = (
    <Button
      id="mephisto-search-button"
      round={true}
      onClick={setFiltersImmediately}
    >
      Search
    </Button>
  );

  if (mode === "OBO") return <Redirect to={`/${data && data.id}`} />;
  return (
    <>
      <Navbar fixedToTop={true}>
        <div className="navbar-wrapper">
          <NavbarGroup className="navbar-header">
            <NavbarHeading>
              <b>
                <pre>mephisto review</pre>
              </b>
            </NavbarHeading>
          </NavbarGroup>
          <NavbarGroup align={Alignment.CENTER}>
            <Tooltip
              content="Separate multiple filters with commas"
              placement={Alignment.LEFT}
            >
              <InputGroup
                id="mephisto-search"
                className="all-item-view-search-bar"
                leftIcon="search"
                onChange={(event) => delaySetFilters(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") setFiltersImmediately();
                }}
                placeholder="Filter data..."
                value={filtersBuffer}
                rightElement={searchButton}
              />
            </Tooltip>
          </NavbarGroup>
        </div>
      </Navbar>
      <main className={`all-item-view mode-${mode}`} id="all-item-view-wrapper">
        <div className="item-dynamic">
          <ErrorPane error={error} />
          {isLoading ? (
            <h1 className="all-item-view-message">Loading...</h1>
          ) : isFinished ? (
            <h1 className="all-item-view-message">
              Done reviewing! You can close this app now
            </h1>
          ) : data && data.length > 0 ? (
            <>
              <CollectionRenderer items={data} itemRenderer={itemRenderer} />
              {pagination && totalPages > 1 ? (
                <Pagination
                  totalPages={totalPages}
                  page={page}
                  setPage={setPage}
                />
              ) : null}
            </>
          ) : (
            <div className="all-item-view-message all-item-view-no-data">
              <h3>
                Thanks for using the <code>$ mephisto review</code> interface.
                Here are a few ways to get started:
              </h3>
              <h3>
                1. Review data from a .csv or{" "}
                <a href="https://jsonlines.org/">.jsonl</a> file
              </h3>
              <pre>
                $ cat sample-data<span className="highlight">.json</span> |
                mephisto review review-app/build/{" "}
                <span className="highlight">--json</span> --all --stdout
              </pre>
              <pre>
                $ cat sample-data<span className="highlight">.csv</span> |
                mephisto review review-app/build/{" "}
                <span className="highlight">--csv</span> --all --stdout
              </pre>
              <h3>2. Review data from the Mephisto database</h3>
              <pre>
                $ mephisto review review-app/build/{" "}
                <span className="highlight">--db mephisto_db_task_name</span>{" "}
                --all --stdout
              </pre>
            </div>
          )}
        </div>
      </main>
    </>
  );
}

export default CollectionView;
