// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import React from "react";
import { useMephistoReview } from "mephisto-review-hook";
import { useParams, useHistory, Link } from "react-router-dom";
import {
  Button,
  Navbar,
  NavbarGroup,
  NavbarDivider,
  NavbarHeading,
  Alignment,
  Position,
  Toaster,
} from "@blueprintjs/core";
import { JSONItem } from "../renderers/JSONItem";
import { getHostname } from "../utils";
import ErrorPane from "./ErrorPane";

const AppToaster = Toaster.create({
  className: "recipe-toaster",
  position: Position.TOP,
});

function ItemView({
  itemRenderer: ItemRenderer = JSONItem,
  wrapClass,
  allowReview = true,
}) {
  const { id } = useParams();
  const {
    data: item,
    isFinished,
    isLoading,
    submit,
    error,
    mode,
  } = useMephistoReview({ taskId: id, hostname: getHostname() });

  const history = useHistory();

  const confirmReview = () => {
    if (mode === "OBO") {
      history.push("/");
    } else {
      AppToaster.show({ message: "Review response recorded." });
    }
  };

  const reviewError = (error) => {
    AppToaster.show({ message: `ERROR: ${error}` });
  };

  const buttonDisable = error || isFinished || isLoading || item == null;

  const hideReviewWorkflow = !allowReview && mode === "ALL";

  return (
    <>
      <Navbar fixedToTop={true}>
        <div className="navbar-wrapper">
          <NavbarGroup>
            {mode === "ALL" ? (
              <>
                <Link to="/" style={{ textDecoration: "none" }}>
                  <Button intent="primary" icon="caret-left" id="home-button">
                    <b>Mephisto Review</b>
                  </Button>
                </Link>
                <NavbarDivider />
              </>
            ) : null}
            <NavbarHeading className="navbar-header">
              {hideReviewWorkflow ? (
                <b>Viewing task:</b>
              ) : (
                <b>Please review the following item:</b>
              )}
            </NavbarHeading>
          </NavbarGroup>
          {hideReviewWorkflow ? null : (
            <NavbarGroup align={Alignment.RIGHT}>
              <Button
                className="btn"
                intent="danger"
                disabled={buttonDisable}
                id="reject-button"
                onClick={async () => {
                  var response = await submit({ result: "rejected" });
                  if (response === "SUCCESS") {
                    confirmReview();
                  } else if (response) {
                    reviewError(response);
                  }
                }}
              >
                <b>REJECT</b>
              </Button>
              <Button
                className="btn"
                intent="success"
                disabled={buttonDisable}
                id="approve-button"
                onClick={async () => {
                  var response = await submit({ result: "approved" });
                  if (response === "SUCCESS") {
                    confirmReview();
                  } else if (response) {
                    reviewError(response);
                  }
                }}
              >
                <b>APPROVE</b>
              </Button>
            </NavbarGroup>
          )}
        </div>
      </Navbar>
      <main className={`item-view mode-${mode}`}>
        {isLoading ? (
          <div className="item-dynamic">
            <ErrorPane error={error} />
            <h1 className="item-view-message">Loading...</h1>
          </div>
        ) : isFinished ? (
          <div className="item-dynamic">
            <ErrorPane error={error} />
            <h1 className="item-view-message">
              Done reviewing! You can close this app now
            </h1>
          </div>
        ) : item ? (
          wrapClass ? (
            <div className={wrapClass}>
              <ErrorPane error={error} />
              <ItemRenderer item={item} />
            </div>
          ) : (
            <>
              <ErrorPane error={error} />
              <ItemRenderer item={item} />
            </>
          )
        ) : (
          <div className="item-dynamic">
            <div className="item-view-message item-view-no-data">
              <ErrorPane error={error} />
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
                <span className="highlight">--json</span> --stdout
              </pre>
              <pre>
                $ cat sample-data<span className="highlight">.csv</span> |
                mephisto review review-app/build/{" "}
                <span className="highlight">--csv</span> --stdout
              </pre>
              <h3>2. Review data from the Mephisto database</h3>
              <pre>
                $ mephisto review review-app/build/{" "}
                <span className="highlight">--db mephisto_db_task_name</span>{" "}
                --stdout
              </pre>
            </div>
          </div>
        )}
      </main>
    </>
  );
}

export default ItemView;
