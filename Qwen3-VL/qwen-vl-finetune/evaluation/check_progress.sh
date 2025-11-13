#!/bin/bash
#
# Check evaluation progress
#

echo "================================"
echo "FLARE25 Evaluation Progress"
echo "================================"
echo ""

# Check if evaluation is running
if pgrep -f "evaluate_flare25.py" > /dev/null; then
    echo "✓ Evaluation is RUNNING"

    # Get process info
    ps aux | grep "evaluate_flare25.py" | grep -v grep | \
        awk '{print "  PID:", $2, "| CPU:", $3"%", "| Mem:", $4"%", "| Time:", $10}'
    echo ""
else
    echo "✗ Evaluation is NOT running"
    echo ""
fi

# Count completed datasets
if [ -d "evaluation_results" ]; then
    COMPLETED=$(ls evaluation_results/*_predictions.json 2>/dev/null | wc -l)
    TOTAL=26

    echo "Completed datasets: $COMPLETED / $TOTAL"

    if [ $COMPLETED -gt 0 ]; then
        echo ""
        echo "Completed files:"
        ls -lth evaluation_results/*_predictions.json | head -10 | \
            awk '{print "  -", $9, "(" $5 ")", $6, $7, $8}'

        if [ $COMPLETED -gt 10 ]; then
            echo "  ... and $((COMPLETED - 10)) more"
        fi
    fi

    echo ""

    # Check if summary exists (evaluation complete)
    if [ -f "evaluation_results/evaluation_summary.json" ]; then
        echo "✓ Evaluation COMPLETE!"
        echo ""
        echo "Summary file:"
        ls -lh evaluation_results/evaluation_summary.json | \
            awk '{print "  -", $9, "(" $5 ")"}'

        echo ""
        echo "Visualization plots:"
        ls evaluation_results/*.png 2>/dev/null | while read file; do
            size=$(ls -lh "$file" | awk '{print $5}')
            echo "  - $(basename "$file") ($size)"
        done
    else
        echo "⏳ Evaluation in progress..."

        # Estimate time remaining
        if [ $COMPLETED -gt 0 ] && pgrep -f "evaluate_flare25.py" > /dev/null; then
            ELAPSED=$(ps -o etimes= -p $(pgrep -f "evaluate_flare25.py" | head -1))
            AVG_TIME=$((ELAPSED / COMPLETED))
            REMAINING=$((TOTAL - COMPLETED))
            EST_REMAINING=$((AVG_TIME * REMAINING))

            echo ""
            echo "Time elapsed: $((ELAPSED / 60)) minutes"
            echo "Average per dataset: $((AVG_TIME / 60)) minutes"
            echo "Estimated remaining: $((EST_REMAINING / 60)) minutes"
        fi
    fi
else
    echo "No results directory found."
    echo "Evaluation has not been started."
fi

echo ""
echo "================================"
