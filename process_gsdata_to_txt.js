/**
 * Process SmartWitness sensor data files (.gsdata, .hdgyro, .gpsdata)
 * and export coordinate points (dateTime, x, y, z) into text files.
 *
 * Node.js implementation based on the user-provided reference parser.
 */

const fs = require("fs");
const path = require("path");
const zlib = require("zlib");

function parseGSData(buffer) {
    const points = [];
    let offset = 0;

    while (offset + 14 <= buffer.length) {
        const epochTime = Number(buffer.readBigInt64LE(offset));
        offset += 8;

        const x = buffer.readInt16LE(offset);
        offset += 2;

        const y = buffer.readInt16LE(offset);
        offset += 2;

        const z = buffer.readInt16LE(offset);
        offset += 2;

        points.push({
            dateTime: epochTime,
            x,
            y,
            z
        });
    }

    return { type: "GS Data", data: points };
}

function parseHDGyro(buffer) {
    const points = [];
    let offset = 0;

    while (offset + 20 <= buffer.length) {
        const epochTime = Number(buffer.readBigInt64LE(offset));
        offset += 8;

        const x = buffer.readInt32LE(offset);
        offset += 4;

        const y = buffer.readInt32LE(offset);
        offset += 4;

        const z = buffer.readInt32LE(offset);
        offset += 4;

        points.push({
            dateTime: epochTime,
            x,
            y,
            z
        });
    }

    return { type: "HD Gyro", data: points };
}

function parseSensorFile(filePath) {
    const compressedBuffer = fs.readFileSync(filePath);
    const result = zlib.gunzipSync(compressedBuffer);
    const lowerFile = filePath.toLowerCase();

    if (lowerFile.endsWith("gpsdata")) {
        return {
            type: "GPS JSON",
            data: JSON.parse(result.toString("utf8")),
        };
    } else if (lowerFile.endsWith("hdgyro")) {
        return parseHDGyro(result);
    } else if (lowerFile.endsWith("gsdata")) {
        return parseGSData(result);
    } else {
        return { message: "Unknown file type" };
    }
}

function formatISOTime(epochMs) {
    try {
        return new Date(epochMs).toISOString();
    } catch {
        return "";
    }
}

function savePointsToTxt(points, outputFilePath, sourceFilename, dataType) {
    const lines = [
        `# Source File: ${sourceFilename}`,
        `# Data Type:   ${dataType}`,
        `# Record Count: ${points.length}`,
        "# Format: dateTime (epoch ms), dateTime (UTC ISO), x, y, z",
        "dateTime,dateTimeISO,x,y,z"
    ];

    for (const pt of points) {
        const iso = formatISOTime(pt.dateTime);
        lines.push(`${pt.dateTime},${iso},${pt.x},${pt.y},${pt.z}`);
    }

    fs.writeFileSync(outputFilePath, lines.join("\n") + "\n", "utf8");
}

function processFolder(inputFolder, outputFolder) {
    if (!fs.existsSync(inputFolder)) {
        console.error(`Error: Input folder '${inputFolder}' does not exist.`);
        return;
    }

    if (!fs.existsSync(outputFolder)) {
        fs.mkdirSync(outputFolder, { recursive: true });
    }

    console.log(`Source folder:      ${inputFolder}`);
    console.log(`Destination folder: ${outputFolder}`);
    console.log("-".repeat(60));

    const files = fs.readdirSync(inputFolder)
        .filter(f => f.toLowerCase().endsWith(".gsdata") || f.toLowerCase().endsWith(".hdgyro") || f.toLowerCase().endsWith(".gpsdata"))
        .sort();

    if (files.length === 0) {
        console.log(`No sensor files found in '${inputFolder}'.`);
        return;
    }

    let processedCount = 0;
    for (const filename of files) {
        const inputPath = path.join(inputFolder, filename);
        const baseName = path.parse(filename).name;
        const outputPath = path.join(outputFolder, `${baseName}.txt`);

        try {
            const parsed = parseSensorFile(inputPath);
            const dataType = parsed.type || "Unknown";
            const data = parsed.data || [];

            if (Array.isArray(data) && (dataType === "GS Data" || dataType === "HD Gyro")) {
                savePointsToTxt(data, outputPath, filename, dataType);
                console.log(`[OK] Processed [${dataType}]: ${filename}`);
                console.log(`     -> Records: ${data.length.toLocaleString()}`);
                console.log(`     -> Saved:   ${outputPath}`);
                if (data.length > 0) {
                    console.log(`     -> First:   ${JSON.stringify(data[0])}`);
                    console.log(`     -> Last:    ${JSON.stringify(data[data.length - 1])}`);
                }
                console.log();
                processedCount++;
            } else {
                fs.writeFileSync(outputPath, JSON.stringify(data, null, 2), "utf8");
                console.log(`[OK] Processed [${dataType}]: ${filename} -> ${outputPath}\n`);
                processedCount++;
            }
        } catch (err) {
            console.error(`[ERROR] Failed to process ${filename}: ${err.message}\n`);
        }
    }

    console.log("-".repeat(60));
    console.log(`Done! Successfully converted ${processedCount} files into '${outputFolder}'.`);
}

// CLI entry point
const inputDir = process.argv[2] || "downloads_gsensor_yes";
const outputDir = process.argv[3] || "downloads_gsensor_yes_txt";
processFolder(inputDir, outputDir);
