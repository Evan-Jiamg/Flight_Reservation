-- 1. 旅程資訊 (Info)
CREATE TABLE
    Info (
        infoId INT AUTO_INCREMENT PRIMARY KEY,
        departureDate DATE NOT NULL,
        -- location 展開為 departure 與 destination
        departure VARCHAR(100) NOT NULL,
        destination VARCHAR(100) NOT NULL,
        
        UNIQUE (departure, destination)
    );

-- 2. 乘客 (Passenger)
CREATE TABLE
    Passenger (
        passengerID INT AUTO_INCREMENT PRIMARY KEY,
        lastName VARCHAR(50) NOT NULL,
        firstName VARCHAR(50) NOT NULL,
        gender CHAR(1) NOT NULL CHECK (gender IN ('M', 'F')),
        birthDate DATE NOT NULL,
        nationality VARCHAR(50) NOT NULL
    );

-- 3. 訂單 (Order)
CREATE TABLE
    `Order` (
        orderID INT AUTO_INCREMENT PRIMARY KEY,
        orderPrice INT NOT NULL CHECK (orderPrice > 0),
        orderDate DATETIME NOT NULL
    );

-- 4. 交易 (Transaction)
CREATE TABLE
    `Transaction` (
        transNo INT AUTO_INCREMENT PRIMARY KEY,
        cardID VARCHAR(50) NOT NULL,
        cardName VARCHAR(100) NOT NULL,
        expiryDate CHAR(5) NOT NULL CHECK (expiryDate REGEXP '^[0-9]{2}/[0-9]{2}$'),
        securityCode CHAR(4) NOT NULL CHECK (securityCode REGEXP '^[0-9]{3,4}$'),
        transTime DATETIME NOT NULL,
        orderID INT UNIQUE NOT NULL,
        FOREIGN KEY (orderID) REFERENCES `Order` (orderID)
        
        -- 確保 [同張卡 + 同時間] → 唯一一筆交易，防止重複提交
        CONSTRAINT unique_card_time UNIQUE (cardID, transTime)
    );

-- 5. 航班 (Itinerary)
CREATE TABLE
    Itinerary (
        itineraryNum INT AUTO_INCREMENT PRIMARY KEY,
        departureTime TIME NOT NULL,
        arrivalDate DATE NOT NULL,
        arrivalTime TIME NOT NULL,
        travelTime TIME NOT NULL,
        departureDate DATE NOT NULL,
        FOREIGN KEY (departureDate) REFERENCES Info (departureDate)
        
        -- 確保實際 [出發 → 抵達] 時間差，等於 總旅行時間 (travelTime)
        CONSTRAINT check_travel_time CHECK (
            TIMESTAMPDIFF (
                MINUTE,
                CONCAT (departureDate, ' ', departureTime),
                CONCAT (arrivalDate, ' ', arrivalTime)
            ) = TIME_TO_SEC (travelTime) / 60
        )
    );

-- 6. 航空公司 (Airline)
CREATE TABLE
    Airline (
        airlineCode VARCHAR(50) PRIMARY KEY,
        airlineName VARCHAR(100) NOT NULL UNIQUE
    );

-- 7. 航段 (flightSegment)
CREATE TABLE
    flightSegment (
        segmentID INT AUTO_INCREMENT PRIMARY KEY,
        departureTime TIME NOT NULL,
        arrivalTime TIME NOT NULL,
        departureAirport VARCHAR(100) NOT NULL,
        arrivalAirport VARCHAR(100) NOT NULL,
        duration TIME NOT NULL,
        airlineCode VARCHAR(50) NOT NULL,
        FOREIGN KEY (airlineCode) REFERENCES Airline (airlineCode),
        
        -- 確保航段 [出發 → 抵達] 時間差，等於 飛行時間 (duration)
        CONSTRAINT check_segment_duration CHECK (
            TIMESTAMPDIFF (MINUTE, departureTime, arrivalTime) = TIME_TO_SEC (duration) / 60
        ),
        UNIQUE (departureAirport, arrivalAirport)
    );

-- 8. 航班_航段 關聯 (Itinerary_flightSegment)
CREATE TABLE
    Itinerary_flightSegment (
        itineraryNum INT,
        segmentID INT,
        PRIMARY KEY (itineraryNum, segmentID),
        FOREIGN KEY (itineraryNum) REFERENCES Itinerary (itineraryNum),
        FOREIGN KEY (segmentID) REFERENCES flightSegment (segmentID)
    );

-- 9. 機票 (Ticket)
CREATE TABLE
    Ticket (
        ticketId INT AUTO_INCREMENT PRIMARY KEY,
        cabinClass VARCHAR(50) NOT NULL CHECK (
            cabinClass IN ('Economy', 'Premium Economy', 'Business', 'First')
        ),
        orderID INT NOT NULL,
        ticketNum INT NOT NULL CHECK (ticketNum > 0),
        ticketTotalPrice INT NOT NULL CHECK (ticketTotalPrice > 0),
        infoId INT NOT NULL,
        passengerID INT NOT NULL,
        itineraryNum INT NOT NULL,
        FOREIGN KEY (orderID) REFERENCES `Order` (orderID),
        FOREIGN KEY (infoId) REFERENCES Info (infoId),
        FOREIGN KEY (passengerID) REFERENCES Passenger (passengerID),
        FOREIGN KEY (itineraryNum) REFERENCES Itinerary (itineraryNum)
    );