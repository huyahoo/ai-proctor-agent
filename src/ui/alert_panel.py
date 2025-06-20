from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QFrame, QScrollArea, QGraphicsDropShadowEffect, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QColor

class EventCard(QFrame):
    """A custom widget to display an event as a card."""
    def __init__(self, event_data: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        timestamp = f"[{event_data.get('timestamp', 0):.2f}s]"
        event_type = event_data.get('type', 'Event').replace('_', ' ').title()
        
        self.timestamp_label = QLabel(timestamp)
        self.type_label = QLabel(event_type)
        self.type_label.setStyleSheet("font-weight: bold;")
        
        layout.addWidget(self.timestamp_label)
        layout.addWidget(self.type_label, 1)
        self.set_selected(False)

    def set_selected(self, selected: bool) -> None:
        if selected:
            self.setStyleSheet("""
                background-color: #06C755;
                color: white;
                border-radius: 8px;
                padding: 2px;
                margin: 1px;
            """)
        else:
            self.setStyleSheet("""
                background-color: #d7d7d7;
                color: #333333;
                border-radius: 8px;
                padding: 2px;
                margin: 1px;
            """)

class AlertPanel(QWidget):
    feedback_provided = pyqtSignal(dict, str)
    analysis_requested = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.active_event_data = None
        self._init_ui()

    def _init_ui(self) -> None:
        self.setStyleSheet("""
            #AlertContentFrame {
                background-color: #6fd283;
                border-radius: 8px;
            }
            QLabel.TitleLabel {
                font-size: 14px;
                font-weight: bold;
                color: #333333;
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 11px;
                padding: 4px 16px;
                max-height: 22px;
            }
            QListWidget#ContentBox, QWidget#ContentBox {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 16px;
                padding-top: 25px;
            }
            QLabel.ContentLabel {
                padding: 10px;
                color: #333333;
                font-size: 14px;
                background-color: transparent;
            }
        """)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        content_frame = self._create_content_frame(outer_layout)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(5)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QColor(215,215,215, 80))
        content_frame.setGraphicsEffect(shadow)

        main_layout = content_frame.layout()
        main_layout.setSpacing(15)

        # --- Section 1: Detected Suspicious Actions ---
        section1_container = QWidget()
        section1_layout = QGridLayout(section1_container)
        section1_layout.setContentsMargins(0, 0, 0, 0)
        section1_layout.setSpacing(0)
        
        anomalies_title = QLabel("Detected Suspicious Actions")
        anomalies_title.setProperty("class", "TitleLabel")
        
        self.event_list_widget = QListWidget()
        self.event_list_widget.setObjectName("ContentBox")
        self.event_list_widget.itemClicked.connect(self._on_event_clicked)
        
        section1_layout.addWidget(anomalies_title, 0, 0, Qt.AlignmentFlag.AlignHCenter)
        section1_layout.addWidget(self.event_list_widget, 1, 0)
        main_layout.addWidget(section1_container, 2)

        # --- Section 2: Monitoring Constraint ---
        section2_container = QWidget()
        section2_layout = QGridLayout(section2_container)
        section2_layout.setContentsMargins(0, 0, 0, 0)
        section2_layout.setSpacing(0)

        constraint_title = QLabel("Monitoring Constraint")
        constraint_title.setProperty("class", "TitleLabel")
        
        constraint_content_box = QWidget()
        constraint_content_box.setObjectName("ContentBox")
        constraint_box_layout = QVBoxLayout(constraint_content_box)
        self.llm_constraint_label = QLabel("<i>No event selected.</i>")
        self.llm_constraint_label.setProperty("class", "ContentLabel")
        self.llm_constraint_label.setWordWrap(True)
        self.llm_constraint_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        constraint_box_layout.addWidget(self.llm_constraint_label)

        section2_layout.addWidget(constraint_title, 0, 0, Qt.AlignmentFlag.AlignHCenter)
        section2_layout.addWidget(constraint_content_box, 1, 0)
        main_layout.addWidget(section2_container, 1)

        # --- Section 3: Proctor Results ---
        section3_container = QWidget()
        section3_layout = QGridLayout(section3_container)
        section3_layout.setContentsMargins(0, 0, 0, 0)
        section3_layout.setSpacing(0)

        results_title = QLabel("Proctor Results")
        results_title.setProperty("class", "TitleLabel")
        
        results_content_box = QWidget()
        results_content_box.setObjectName("ContentBox")
        results_box_layout = QVBoxLayout(results_content_box)
        self.vlm_decision_label = QLabel("<i>No event selected.</i>")
        self.vlm_decision_label.setProperty("class", "ContentLabel")
        self.vlm_decision_label.setWordWrap(True)
        self.vlm_decision_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        results_box_layout.addWidget(self.vlm_decision_label)
        
        section3_layout.addWidget(results_title, 0, 0, Qt.AlignmentFlag.AlignHCenter)
        section3_layout.addWidget(results_content_box, 1, 0)
        main_layout.addWidget(section3_container, 2)
        
        self._init_feedback_buttons(main_layout)
        self.clear_all_events()

    def add_event(self, event_data: dict) -> None:
        card = EventCard(event_data)
        item = QListWidgetItem(self.event_list_widget)
        item.setSizeHint(QSize(0, 50))
        item.setData(Qt.ItemDataRole.UserRole, event_data)
        self.event_list_widget.addItem(item)
        self.event_list_widget.setItemWidget(item, card)
        
    def _on_event_clicked(self, item: QListWidgetItem) -> None:
        for i in range(self.event_list_widget.count()):
            list_item = self.event_list_widget.item(i)
            card = self.event_list_widget.itemWidget(list_item)
            card.set_selected(list_item == item)
        
        event_data = item.data(Qt.ItemDataRole.UserRole)
        self.active_event_data = event_data
        
        self.llm_constraint_label.setText("<i>Requesting AI analysis...</i>")
        self.vlm_decision_label.setText("<i>Waiting for LLM constraints...</i>")
        self.analysis_requested.emit(event_data)

    def update_llm_constraint(self, constraint: str) -> None:
        if self.active_event_data: self.llm_constraint_label.setText(constraint)

    def update_vlm_result(self, data: dict) -> None:
        event_data = data.get('event_data', {})
        if self.active_event_data and self.active_event_data.get('event_id') == event_data.get('event_id'):
            vlm_result = data.get('vlm_result', {})
            decision = vlm_result.get('decision', 'N/A')
            explanation = vlm_result.get('explanation', '')
            
            color = "#333333"
            if "Confirmed" in decision: color = "#e74c3c"
            elif "Not Cheating" in decision: color = "#4CC764"
            
            self.vlm_decision_label.setText(f"<span style='color:{color}; font-weight:bold;'>{decision}</span><br>{explanation}")
            self.active_event_data.update(vlm_decision_text=decision, vlm_explanation_text=explanation)

    def clear_all_events(self) -> None:
        self.event_list_widget.clear()
        self.llm_constraint_label.setText("<i>No event selected.</i>")
        self.vlm_decision_label.setText("<i>No event selected.</i>")
        self.active_event_data = None
        
    # --- UI Creation Helpers ---
    def _create_content_frame(self, parent_layout: QVBoxLayout) -> QFrame:
        content_frame = QFrame()
        content_frame.setObjectName("AlertContentFrame")
        layout = QVBoxLayout(content_frame)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        parent_layout.addWidget(content_frame)
        return content_frame

    def _init_feedback_buttons(self, parent_layout: QVBoxLayout) -> None:
        feedback_layout = QHBoxLayout()
        self.confirm_btn = QPushButton("✔️ Confirm Cheating")
        self.false_pos_btn = QPushButton("❌ False Positive")
        self.confirm_btn.setStyleSheet("""
            background-color: #e74c3c;
            padding: 10px 24px; font-size: 14px; margin: 4px 2px;
            border-radius: 8px; font-weight: bold;
        """)
        self.false_pos_btn.setStyleSheet("""
            background-color: #3498db;
            padding: 10px 24px; font-size: 14px; margin: 4px 2px;
            border-radius: 8px; font-weight: bold;
        """)
        self.confirm_btn.clicked.connect(lambda: self.emit_feedback("confirmed_cheating"))
        self.false_pos_btn.clicked.connect(lambda: self.emit_feedback("false_positive"))
        feedback_layout.addWidget(self.confirm_btn)
        feedback_layout.addWidget(self.false_pos_btn)
        parent_layout.addLayout(feedback_layout)
        # Hide the buttons as requested
        self.confirm_btn.setVisible(False)
        self.false_pos_btn.setVisible(False)

    def emit_feedback(self, feedback_type: str) -> None:
        if self.active_event_data:
            self.feedback_provided.emit(self.active_event_data, feedback_type)

