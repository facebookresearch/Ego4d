// Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import React from "react";
import ReactPlayer from "react-player"; /* dependency */
import { getHostname } from "../utils";
import { Callout, Intent } from "@blueprintjs/core";

import "./NarrationsApp.css";
const USE_MOCK = false;

const samplePayloadForInstructions = {
  file: "https://video.cdn.net/sample_url_a3cda33df8.mp4",
  info: {
    type: "TIME_SEGMENTATION",
    role: "RESULT",
    payload: [
      {
        start_time: 168.032,
        end_time: 168.032,
        label:
          "C was in a kitchen. C cooked in a pot with a wooden ladle. C sliced some veggies with a knife. #Summary. ",
      },
      {
        start_time: 0.70267,
        end_time: 0.70267,
        label: "#C C moves a nylon by a kitchen rack with his hands ",
      },
    ],
  },
  taxonomy: "narration",
};

function mapToTimePoint(seg, useTimePoint) {
  if (!useTimePoint) {
    return seg;
  }
  const { start_time, end_time, ...rest } = seg;
  return { start_time, end_time: start_time + 1, ...rest };
}

function App({ data }) {
  return <Viewer data={data} />;
}

function parsePayload(cb) {
  let parsedData;
  try {
    let input = prompt(
      "Enter a payload to view:\n(Ensure formatting is correct, e.g. all keys are enclosed with quotes)\n\nSample:\n" +
        JSON.stringify(samplePayloadForInstructions, null, 2)
    );
    parsedData = JSON.parse(input);
    console.log("data parsed!", parsedData);
  } catch (e) {
    console.log("data parse failed!");
    alert(JSON.stringify(e.toString()));
    parsedData = undefined;
  }
  cb(parsedData);
}

function Viewer({ data }) {
  //   const [taskData, setTaskData] = React.useState(undefined);
  //   const fullMock = {
  //     data: taskData || {
  //       file: "https://interncache-rva.fbcdn.net/v/t53.39266-7/10000000_206988614366921_5057087031069697290_n.mp4?ccb=2&_nc_sid=5f5f54&efg=eyJ1cmxnZW4iOiJwaHBfdXJsZ2VuX2NsaWVudC9pbnRlcm4vc2l0ZS94L2ZiY2RuIn0%3D&_nc_ht=interncache-rva&_nc_rmd=260&oh=0bcbce9155e15ed4d9a16a904bf125ae&oe=603C6521",
  //       info: mockData,
  //     },
  //     isFinished: false,
  //     isLoading: false,
  //   };
  //   let { data, isFinished, isLoading, submit, error } = useMephistoReview({
  //     useMock: USE_MOCK || !!taskData,
  //     mock: fullMock,
  //   });

  const file = getHostname() + data?.file;
  const useTimePoint = data?.taxonomy === "narration";
  const segData =
    data?.info?.payload.map((d) => mapToTimePoint(d, useTimePoint)) || [];

  const [progress, setProgress] = React.useState(null);
  const [duration, setDuration] = React.useState(null);
  const [playing, setPlaying] = React.useState(false);
  const vidRef = React.useRef();

  const [isError, setError] = React.useState(false);

  const activeAnnotations = segData
    .filter(
      (seg) =>
        progress >= seg.start_time - 0.5 && progress <= seg.end_time + 0.5
    )
    .map((seg) => (
      <Segment
        segment={seg}
        duration={duration}
        progress={progress}
        onClick={() => {
          vidRef?.current && vidRef.current.seekTo(seg.start_time, "seconds");
          setPlaying(true);
        }}
      />
    ));

  return (
    <div>
      <div className="app-container">
        <div className="video-viewer">
          {isError ? (
            <Callout intent={Intent.WARNING} style={{ marginBottom: 10 }}>
              The video was not found. You may not have it downloaded. You can
              try downloading it with the Ego4D cli:
              <pre style={{ whiteSpace: "break-spaces" }}>
                python -m ego4d.cli.cli --yes --datasets full_scale
                --output_directory $OUTPUT_DIR --video_uids {data.uid}
              </pre>
            </Callout>
          ) : null}
          <ReactPlayer
            url={file}
            controls
            playing={playing}
            ref={vidRef}
            width={"100%"}
            progressInterval={350}
            onError={(error) => {
              console.log(error);
              setError(true);
            }}
            onProgress={({ playedSeconds }) => {
              setProgress(playedSeconds);
            }}
            onDuration={setDuration}
          />
          {/* <h3>{Math.floor(progress * 10) / 10}</h3>
            <h3>{duration}</h3> */}
          <h3>Active annotations:</h3>
          {activeAnnotations.length > 0 ? activeAnnotations : <span>None</span>}
        </div>
        <div className="segment-viewer">
          <h3>All annotations:</h3>
          {segData.map((seg) => (
            <Segment
              useTimePoint={useTimePoint}
              segment={seg}
              duration={duration}
              isActive={progress >= seg.start_time && progress <= seg.end_time}
              onClick={() => {
                vidRef?.current &&
                  vidRef.current.seekTo(seg.start_time, "seconds");
                setPlaying(true);
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function Segment({
  segment,
  onClick,
  duration,
  isActive = false,
  useTimePoint,
}) {
  return (
    <div
      className={"segment-wrapper " + (isActive ? "active" : "inactive")}
      onClick={onClick}
      onKeyDown={onClick}
      role="button"
      tabIndex={0}
    >
      {useTimePoint ? (
        <div className="duration">
          <span>
            {Math.floor(segment.start_time / 60)}:
            {(segment.start_time % 60).toFixed(0).padStart(2, "0")}
          </span>
        </div>
      ) : (
        <div className="duration">
          <span>
            {Math.floor(segment.start_time / 60)}:
            {(segment.start_time % 60).toFixed(0).padStart(2, "0")}
          </span>{" "}
          &mdash;{" "}
          <span>
            {Math.floor(segment.end_time / 60)}:
            {(segment.end_time % 60).toFixed(0).padStart(2, "0")}
          </span>
          &nbsp;({(segment.end_time - segment.start_time).toFixed(1)}s)
        </div>
      )}
      <div className="track">
        {duration && (
          <div
            className="bar"
            style={{
              width:
                (100 * (segment.end_time - segment.start_time)) / duration +
                "%",
              marginLeft: (100 * segment.start_time) / duration + "%",
            }}
          ></div>
        )}
      </div>
      <div className="segment">{segment.label}</div>
      {segment.tags?.length > 0 ? (
        <div
          className="segment"
          style={{ marginTop: 10, fontSize: 13, fontFamily: "monospace" }}
        >
          {segment.tags.map((s, idx) => (
            <span
              key={idx}
              style={{
                backgroundColor: "#ccc",
                border: "1px solid #aaa",
                borderRadius: 3,
                display: "inline-block",
                marginRight: 5,
                marginBottom: 5,
                padding: 3,
              }}
            >
              {s || <span>&nbsp;</span>}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export default App;
